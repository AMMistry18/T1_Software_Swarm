"""Verify package installation and headless physics on a fresh environment."""

import numpy as np
import pytest

import tar_swarm
from tar_swarm.mujoco.simulation import run_smoke


def test_package_imports():
    assert tar_swarm.__name__ == "tar_swarm"


def test_mujoco_initializes_and_steps():
    model, data = run_smoke()
    assert model.nq == 7
    assert data.time == pytest.approx(10 * model.opt.timestep)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert data.qpos[2] < 1.0
    assert data.qvel[2] < 0.0
