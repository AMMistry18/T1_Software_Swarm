"""MaskablePPO training with checkpoints, resume, and TensorBoard logging.

A short smoke run proves the pipeline executes. It does not prove learning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from tar_swarm.mission_planning.checkpointing import (
    assert_compatible,
    build_checkpoint_meta,
    load_meta,
    save_meta,
)
from tar_swarm.mission_planning.env import MissionPlanningEnv
from tar_swarm.mission_planning.preprocess import prepare_scenario
from tar_swarm.mission_planning.rewards import RewardWeights
from tar_swarm.mission_planning.scenarios import (
    concave_region,
    multi_mapper,
    one_mapper_small,
)
from tar_swarm.mission_planning.schemas import MissionRequest


def _mask_fn(env):
    # ActionMasker may pass wrappers; masks live on MissionPlanningEnv.
    return env.unwrapped.action_masks()


def _curriculum_factory(rng):
    builders = [one_mapper_small, multi_mapper, concave_region]
    req = builders[int(rng.integers(0, len(builders)))]()
    return prepare_scenario(req)


def train_maskable_ppo(
    request: MissionRequest,
    output_dir: str | Path,
    *,
    total_timesteps: int = 10_000,
    n_envs: int = 1,
    seed: int = 0,
    resume_path: str | Path | None = None,
    use_curriculum: bool = True,
    reward_weights: RewardWeights | None = None,
    learning_rate: float = 3e-4,
    device: str = "cpu",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tb"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights = reward_weights or RewardWeights()
    train_config = {
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "use_curriculum": use_curriculum,
        "learning_rate": learning_rate,
        "norm_scale_m": 200.0,
        "reward_weights": weights.__dict__,
        "algo": "MaskablePPO",
    }

    def make_train_env(rank: int):
        def _init():
            env = MissionPlanningEnv(
                request=request,
                reward_weights=weights,
                scenario_factory=_curriculum_factory if use_curriculum else None,
            )
            env = ActionMasker(env, _mask_fn)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env

        return _init

    # DummyVecEnv keeps smoke/CI reliable; scale n_envs for throughput.
    vec = DummyVecEnv([make_train_env(i) for i in range(max(1, n_envs))])

    eval_req = one_mapper_small()

    def make_eval_env():
        env = MissionPlanningEnv(request=eval_req, reward_weights=weights)
        env = ActionMasker(env, _mask_fn)
        return Monitor(env)

    eval_env = DummyVecEnv([make_eval_env])

    if resume_path is not None:
        resume_path = Path(resume_path)
        meta = load_meta(resume_path.with_name(resume_path.stem + ".meta.json"))
        assert_compatible(meta, request)
        model = MaskablePPO.load(str(resume_path), env=vec, device=device)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec,
            verbose=0,
            seed=seed,
            learning_rate=learning_rate,
            n_steps=64,
            batch_size=64,
            gamma=0.99,
            tensorboard_log=str(tb_dir),
            device=device,
        )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best"),
        log_path=str(ckpt_dir / "eval"),
        eval_freq=max(256 // max(n_envs, 1), 1),
        n_eval_episodes=1,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_cb,
        progress_bar=False,
        reset_num_timesteps=resume_path is None,
    )
    elapsed = time.time() - t0

    final_path = ckpt_dir / "final_model"
    model.save(str(final_path))
    meta = build_checkpoint_meta(
        request,
        train_config=train_config,
        best_metric=float(getattr(eval_cb, "best_mean_reward", 0.0) or 0.0),
    )
    meta["wall_time_s"] = elapsed
    meta["note"] = (
        "Smoke/short runs prove the pipeline runs; they do not prove learning."
    )
    save_meta(Path(str(final_path) + ".meta.json"), meta)
    best_dir = ckpt_dir / "best"
    if (best_dir / "best_model.zip").exists():
        save_meta(best_dir / "best_model.meta.json", meta)

    summary = {
        "output_dir": str(output_dir),
        "final_model": str(final_path) + ".zip",
        "wall_time_s": elapsed,
        "total_timesteps": total_timesteps,
        "meta": meta,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    vec.close()
    eval_env.close()
    return summary
