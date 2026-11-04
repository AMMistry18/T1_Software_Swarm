"""CLI for preprocess, train, resume, evaluate, plan, validate, replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tar_swarm.mission_planning.api import plan_mission
from tar_swarm.mission_planning.baselines import (
    search_relay_baseline,
    static_relay_baseline,
)
from tar_swarm.mission_planning.compile import compile_from_sim
from tar_swarm.mission_planning.evaluate import evaluate_baselines, evaluate_checkpoint
from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.replay import animate_replay, save_replay
from tar_swarm.mission_planning.scenarios import build_named, load_request_yaml
from tar_swarm.mission_planning.schemas import MissionPlan, MissionRequest
from tar_swarm.mission_planning.train import train_maskable_ppo
from tar_swarm.mission_planning.validate import MissionExecutor, validate_mission


def _load_request(args) -> MissionRequest:
    if args.scenario:
        return build_named(args.scenario)
    if args.request:
        return load_request_yaml(args.request)
    raise SystemExit("Provide --scenario NAME or --request PATH")


def cmd_preprocess(args):
    req = _load_request(args)
    scenario = prepare_scenario(req)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if scenario.warnings:
        for w in scenario.warnings:
            print(f"warning: {w}")


def cmd_train(args):
    req = _load_request(args)
    summary = train_maskable_ppo(
        req,
        args.output,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        resume_path=args.resume,
        use_curriculum=not args.no_curriculum,
    )
    keys = ("output_dir", "final_model", "wall_time_s", "total_timesteps")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))


def cmd_evaluate(args):
    req = _load_request(args)
    if args.checkpoint:
        result = evaluate_checkpoint(req, args.checkpoint, args.output)
    else:
        result = {"baselines": evaluate_baselines(req)}
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def cmd_plan(args):
    req = _load_request(args)
    result = plan_mission(req, args.checkpoint, fallback=args.fallback)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"success={result.success} source={result.source} -> {out}")
    if not result.success:
        print(result.failure.message)
        sys.exit(2)


def cmd_validate(args):
    req = _load_request(args)
    scenario = prepare_scenario(req)
    plan = MissionPlan.model_validate_json(Path(args.plan).read_text(encoding="utf-8"))
    report = validate_mission(scenario, plan, timing_jitter_s=args.jitter)
    payload = {
        "ok": report.ok,
        "coverage_fraction": report.coverage_fraction,
        "disconnected_mapper_s": report.disconnected_mapper_s,
        "separation_violations": report.separation_violations,
        "flight_budget_violations": report.flight_budget_violations,
        "dependency_failures": report.dependency_failures,
        "details": report.details,
        "assumptions_established": report.assumptions_established,
        "requires_real_measurement": report.requires_real_measurement,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not report.ok:
        sys.exit(2)


def cmd_replay(args):
    req = _load_request(args)
    scenario = prepare_scenario(req)
    if args.baseline == "static":
        base = static_relay_baseline(scenario)
    else:
        base = search_relay_baseline(scenario)
    paths = save_replay(
        scenario,
        base.sim,
        args.output,
        save_gif=bool(args.gif or args.animate),
    )
    plan = compile_from_sim(scenario, base.sim, base.command_trace)
    plan_path = Path(args.output) / "plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    executor = MissionExecutor(scenario, plan)
    report = executor.run()
    print(
        json.dumps(
            {"replay": paths, "plan": str(plan_path), "validate_ok": report.ok},
            indent=2,
        )
    )
    if args.animate:
        # Live window; close it to return. GIF was also saved above.
        animate_replay(scenario, base.sim, show=True, save_path=None)


def cmd_smoke(args):
    """End-to-end smoke: train, reload, plan, export, validate, replay."""
    from tar_swarm.mission_planning.scenarios import one_mapper_small

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    req = one_mapper_small()
    # Prefer baseline for guaranteed artifact; also smoke-train RL briefly.
    summary = train_maskable_ppo(
        req,
        out / "train",
        total_timesteps=args.timesteps,
        n_envs=1,
        seed=0,
        use_curriculum=False,
    )
    ckpt = summary["final_model"]
    # Plan with baseline search (explicit), then also attempt RL.
    baseline_result = plan_mission(req, checkpoint=None, fallback="static")
    if not baseline_result.success:
        baseline_result = plan_mission(req, checkpoint=None, fallback="search")
    (out / "baseline_plan.json").write_text(
        baseline_result.model_dump_json(indent=2), encoding="utf-8"
    )
    rl_result = plan_mission(req, checkpoint=ckpt)
    (out / "rl_plan.json").write_text(
        rl_result.model_dump_json(indent=2), encoding="utf-8"
    )

    scenario = prepare_scenario(req)
    if baseline_result.success and baseline_result.plan is not None:
        report = validate_mission(scenario, baseline_result.plan)
        base = static_relay_baseline(scenario)
        paths = save_replay(scenario, base.sim, out / "replay")
        if baseline_result.plan is not None:
            (out / "exported_plan.json").write_text(
                baseline_result.plan.model_dump_json(indent=2), encoding="utf-8"
            )
    else:
        report = None
        paths = {}
    print(
        json.dumps(
            {
                "train": summary["final_model"],
                "baseline_success": baseline_result.success,
                "rl_success": rl_result.success,
                "validate_ok": None if report is None else report.ok,
                "replay": paths,
                "note": (
                    "Smoke test proves the pipeline runs; "
                    "it does not prove learning."
                ),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tar-plan", description="TAR mission planning CLI"
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_req(sp):
        sp.add_argument("--scenario", choices=[
            "one_mapper_small", "multi_mapper", "concave_region", "failure_case"
        ])
        sp.add_argument("--request", type=str, help="MissionRequest YAML/JSON path")

    sp = sub.add_parser(
        "preprocess", help="Run deterministic preprocessing"
    )
    add_req(sp)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("train", help="Train MaskablePPO")
    add_req(sp)
    sp.add_argument("--output", required=True)
    sp.add_argument("--timesteps", type=int, default=10_000)
    sp.add_argument("--n-envs", type=int, default=1)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--resume", default=None)
    sp.add_argument("--no-curriculum", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Evaluate policy and/or baselines")
    add_req(sp)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("plan", help="plan_mission API")
    add_req(sp)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--fallback", choices=["static", "search"], default=None)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("validate", help="Validate an exported MissionPlan")
    add_req(sp)
    sp.add_argument("--plan", required=True)
    sp.add_argument("--jitter", type=float, default=0.0)
    sp.add_argument("--output", default=None)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("replay", help="Baseline replay with plots")
    add_req(sp)
    sp.add_argument("--baseline", choices=["static", "search"], default="search")
    sp.add_argument("--output", required=True)
    sp.add_argument(
        "--gif",
        action="store_true",
        help="Also write replay.gif (portable animation)",
    )
    sp.add_argument(
        "--animate",
        action="store_true",
        help="Open a live Matplotlib window and write replay.gif",
    )
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("smoke", help="End-to-end smoke: train/plan/validate/replay")
    sp.add_argument("--output", required=True)
    sp.add_argument("--timesteps", type=int, default=512)
    sp.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
