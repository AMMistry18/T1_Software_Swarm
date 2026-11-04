"""Policy and baseline evaluation on held-out scenarios."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from tar_swarm.mission_planning.baselines import (
    search_relay_baseline,
    static_relay_baseline,
)
from tar_swarm.mission_planning.checkpointing import assert_compatible, load_meta
from tar_swarm.mission_planning.compile import compile_from_sim
from tar_swarm.mission_planning.env import MissionPlanningEnv
from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.schemas import MissionRequest, PlanningResult


def _mask_fn(env):
    return env.action_masks()


def evaluate_baselines(request: MissionRequest) -> dict[str, Any]:
    scenario = prepare_scenario(request)
    t0 = time.time()
    static = static_relay_baseline(scenario)
    t_static = time.time() - t0
    t1 = time.time()
    search = search_relay_baseline(scenario)
    t_search = time.time() - t1
    return {
        "static_relay": {
            **static.metrics,
            "success": static.success,
            "runtime_s": t_static,
        },
        "search_relay": {
            **search.metrics,
            "success": search.success,
            "runtime_s": t_search,
        },
    }


def rollout_policy(
    request: MissionRequest,
    checkpoint: str | Path,
    *,
    deterministic: bool = True,
    max_steps: int = 500,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    meta_path = checkpoint.with_suffix(".meta.json")
    if not meta_path.exists():
        # SB3 saves as path.zip; meta may be path.meta.json beside stem.
        alt = checkpoint.parent / (checkpoint.stem + ".meta.json")
        if checkpoint.suffix == ".zip":
            alt = checkpoint.with_name(checkpoint.stem + ".meta.json")
        meta_path = alt
    meta = load_meta(meta_path)
    assert_compatible(meta, request)

    env = MissionPlanningEnv(request=request)
    env = ActionMasker(env, _mask_fn)
    env = Monitor(env)
    model = MaskablePPO.load(str(checkpoint), env=env)
    obs, _ = env.reset()
    trace = []
    t0 = time.time()
    for _ in range(max_steps):
        if hasattr(env, "action_masks"):
            masks = env.action_masks()
        else:
            masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        # Record decoded command from unwrapped layout.
        raw = env.unwrapped if hasattr(env, "unwrapped") else env
        while hasattr(raw, "env"):
            raw = raw.env
        kind, i0, i1 = raw.layout.decode_action(int(action))
        cmd: dict[str, Any] = {"cmd": kind}
        if kind == "relay":
            cmd["relay"] = raw._relay_ids[i0]
            cmd["candidate"] = raw._candidate_ids[i1]
        elif kind == "release":
            cmd["mapper"] = raw._mapper_ids[i0]
        trace.append(cmd)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    runtime = time.time() - t0
    sim = raw.sim
    metrics = {
        "coverage_fraction": sim.metrics.coverage_fraction,
        "disconnected_mapper_s": sim.metrics.disconnected_mapper_s,
        "relay_travel_m": sim.metrics.relay_travel_m,
        "relay_repositions": sim.metrics.relay_repositions,
        "mission_duration_s": sim.metrics.elapsed_s,
        "mapper_waiting_s": sum(
            d.waiting_s for d in sim.drones.values() if d.role == "mapper"
        ),
        "flight_budget_violations": sim.metrics.flight_budget_violations,
        "separation_violations": sim.metrics.separation_violations,
        "mission_complete": sim.metrics.mission_complete,
        "success": bool(sim.metrics.mission_complete),
        "runtime_s": runtime,
        "term_reason": sim.term_reason,
    }
    return {
        "metrics": metrics,
        "trace": trace,
        "sim": sim,
        "scenario": prepare_scenario(request),
        "mean_reward_eval": None,
    }


def evaluate_checkpoint(
    request: MissionRequest,
    checkpoint: str | Path,
    output_path: str | Path | None = None,
    n_eval_episodes: int = 3,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    env = ActionMasker(Monitor(MissionPlanningEnv(request=request)), _mask_fn)
    model = MaskablePPO.load(str(checkpoint), env=env)
    mean_r, std_r = evaluate_policy(
        model, env, n_eval_episodes=n_eval_episodes, deterministic=True
    )
    roll = rollout_policy(request, checkpoint)
    baselines = evaluate_baselines(request)
    result = {
        "policy": {
            **roll["metrics"],
            "mean_reward": float(mean_r),
            "std_reward": float(std_r),
        },
        "baselines": baselines,
        "note": (
            "Do not assume RL outperforms baselines without measured results. "
            "Smoke training does not prove learning."
        ),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def planning_result_from_rollout(
    request: MissionRequest, checkpoint: str | Path
) -> PlanningResult:
    from tar_swarm.mission_planning.validate import validate_mission

    roll = rollout_policy(request, checkpoint)
    scenario = roll["scenario"]
    sim = roll["sim"]
    plan = compile_from_sim(scenario, sim, roll["trace"])
    report = validate_mission(scenario, plan)
    plan = plan.model_copy(update={"validation": {
        "ok": report.ok,
        "coverage_fraction": report.coverage_fraction,
        "disconnected_mapper_s": report.disconnected_mapper_s,
        "separation_violations": report.separation_violations,
        "flight_budget_violations": report.flight_budget_violations,
        "dependency_failures": report.dependency_failures,
    }})
    if not report.ok:
        from tar_swarm.mission_planning.schemas import FailureDiagnostics

        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="validation_failed",
                message="Exported mission failed independent validation",
                details=plan.validation,
            ),
            metrics=roll["metrics"],
            source="rl",
        )
    if not roll["metrics"]["success"]:
        from tar_swarm.mission_planning.schemas import FailureDiagnostics

        return PlanningResult(
            success=False,
            failure=FailureDiagnostics(
                code="rollout_incomplete",
                message="RL rollout did not achieve a validated complete mission",
                details=roll["metrics"],
            ),
            metrics=roll["metrics"],
            source="rl",
        )
    return PlanningResult(
        success=True, plan=plan, metrics=roll["metrics"], source="rl"
    )
