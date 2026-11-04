"""Offline RL mission planning for mapper/relay UAV surveys.

This package plans coordinated missions before deployment. Real flight control
and radio behavior remain outside its scope. See docs/mission_planning.md.
"""

from tar_swarm.mission_planning.api import plan_mission
from tar_swarm.mission_planning.schemas import (
    MissionPlan,
    MissionRequest,
    PlanningResult,
    PreparedScenario,
)
from tar_swarm.mission_planning.versions import PREPROCESS_VERSION, SCHEMA_VERSION

__all__ = [
    "MissionPlan",
    "MissionRequest",
    "PlanningResult",
    "PreparedScenario",
    "plan_mission",
    "PREPROCESS_VERSION",
    "SCHEMA_VERSION",
]
