"""Library API: plan_mission(request, checkpoint) -> PlanningResult."""

from __future__ import annotations

from pathlib import Path

from tar_swarm.mission_planning.baselines import (
    search_relay_baseline,
    static_relay_baseline,
)
from tar_swarm.mission_planning.checkpointing import assert_compatible, load_meta
from tar_swarm.mission_planning.compile import compile_from_sim
from tar_swarm.mission_planning.evaluate import planning_result_from_rollout
from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.schemas import (
    FailureDiagnostics,
    MissionRequest,
    PlanningResult,
)
from tar_swarm.mission_planning.validate import validate_mission


def plan_mission(
    request: MissionRequest | dict,
    checkpoint: str | Path | None = None,
    *,
    fallback: str | None = None,
) -> PlanningResult:
    """Plan a coordinated mission.

    Parameters
    ----------
    request:
        MissionRequest or dict matching the schema.
    checkpoint:
        Path to a MaskablePPO checkpoint (.zip). Required for RL planning.
    fallback:
        Optional ``\"search\"`` or ``\"static\"`` baseline used only when
        explicitly requested after RL failure. Never silently substituted
        while attributing success to RL.

    Returns
    -------
    PlanningResult with a validated plan or structured failure diagnostics.
    """
    if isinstance(request, dict):
        request = MissionRequest.model_validate(request)
    else:
        # Re-validate for frozen copies / version checks.
        request = MissionRequest.model_validate(request.model_dump())

    try:
        scenario = prepare_scenario(request)
    except ValueError as exc:
        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="preprocess_rejected",
                message=str(exc),
                details={},
            ),
            source="unknown",
        )

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        meta_candidates = [
            checkpoint.with_suffix(".meta.json"),
            checkpoint.with_name(checkpoint.stem + ".meta.json"),
            checkpoint.parent / "best_model.meta.json",
        ]
        meta = None
        for path in meta_candidates:
            if path.exists():
                meta = load_meta(path)
                break
        if meta is None:
            return PlanningResult(
                success=False,
                failure=FailureDiagnostics(
                    code="missing_checkpoint_meta",
                    message="Checkpoint metadata (.meta.json) is required",
                    details={"checkpoint": str(checkpoint)},
                ),
                source="rl",
            )
        try:
            assert_compatible(meta, request)
        except ValueError as exc:
            return PlanningResult(
                success=False,
                failure=FailureDiagnostics(
                    code="incompatible_checkpoint",
                    message=str(exc),
                    details={"checkpoint": str(checkpoint)},
                ),
                source="rl",
            )
        result = planning_result_from_rollout(request, checkpoint)
        if result.success or fallback is None:
            return result
        # Explicit fallback only — never attributed to RL.
        return _baseline_plan(scenario, fallback)

    if fallback is None:
        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="no_checkpoint",
                message=(
                    "checkpoint is required for RL planning; "
                    "pass fallback= to use a baseline"
                ),
                details={},
            ),
            source="unknown",
        )
    return _baseline_plan(scenario, fallback)


def _baseline_plan(scenario, name: str) -> PlanningResult:
    if name == "static":
        base = static_relay_baseline(scenario)
        source = "baseline_static"
    elif name == "search":
        base = search_relay_baseline(scenario)
        source = "baseline_search"
    else:
        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="unknown_fallback",
                message=f"Unknown fallback {name}",
                details={},
            ),
            source="unknown",
        )
    plan = compile_from_sim(scenario, base.sim, base.command_trace)
    report = validate_mission(scenario, plan)
    plan = plan.model_copy(
        update={
            "validation": {
                "ok": report.ok,
                "coverage_fraction": report.coverage_fraction,
                "disconnected_mapper_s": report.disconnected_mapper_s,
                "separation_violations": report.separation_violations,
                "flight_budget_violations": report.flight_budget_violations,
                "dependency_failures": report.dependency_failures,
            }
        }
    )
    if not (base.success and report.ok):
        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="baseline_failed",
                message="Baseline did not produce a validated mission",
                details={
                    "baseline_metrics": base.metrics,
                    "validation": plan.validation,
                },
            ),
            metrics=base.metrics,
            source=source,
        )
    return PlanningResult(
        success=True, plan=plan, metrics=base.metrics, source=source
    )
