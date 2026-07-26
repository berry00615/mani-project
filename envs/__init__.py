"""
Custom ManiSkill environments for the PPO project.

Importing this package registers custom environments with gymnasium.
"""

from .pick_cube_collision_penalty import PickCubeCollisionPenaltyEnv
from .pick_cube_collision_gripper import PickCubeCollisionGripperEnv
from .pick_cube_gripper_curriculum import PickCubeGripperCurriculumEnv
from .pick_cube_transport_curriculum import PickCubeTransportCurriculumEnv
from .pick_cube_lift_curriculum import PickCubeLiftCurriculumEnv

from .pick_cube_stable_lift_curriculum import PickCubeStableLiftCurriculumEnv

from .pick_cube_target_transport_curriculum import PickCubeTargetTransportCurriculumEnv
from .pick_cube_directed_transport_curriculum import PickCubeDirectedTransportCurriculumEnv
from .pick_cube_goal_brake_curriculum import PickCubeGoalBrakeCurriculumEnv
from .pick_cube_precision_carry_curriculum import PickCubePrecisionCarryCurriculumEnv
from .pick_cube_posture_stable_carry import PickCubePostureStableCarryEnv
from .pick_cube_center_precision_carry import PickCubeCenterPrecisionCarryEnv
from .pick_cube_efficient_center_carry import PickCubeEfficientCenterCarryEnv
from .stack_cube_transport_curriculum import StackCubeTransportCurriculumEnv
from .stack_cube_local_transport_curriculum import StackCubeLocalTransportCurriculumEnv
from .stack_cube_auto_release_curriculum import StackCubeAutoReleaseCurriculumEnv
from .stack_cube_stable_release_curriculum import StackCubeStableReleaseCurriculumEnv

__all__ = [
    "PickCubeCollisionPenaltyEnv",
    "PickCubeCollisionGripperEnv",
    "PickCubeGripperCurriculumEnv",
    "PickCubeTransportCurriculumEnv",
    "PickCubeLiftCurriculumEnv",
    "PickCubeStableLiftCurriculumEnv",
    "PickCubeTargetTransportCurriculumEnv",
    "PickCubeDirectedTransportCurriculumEnv",
    "PickCubeGoalBrakeCurriculumEnv",
    "PickCubePrecisionCarryCurriculumEnv",
    "PickCubePostureStableCarryEnv",
    "PickCubeCenterPrecisionCarryEnv",
    "PickCubeEfficientCenterCarryEnv",
    "StackCubeTransportCurriculumEnv",
    "StackCubeLocalTransportCurriculumEnv",
    "StackCubeAutoReleaseCurriculumEnv",
    "StackCubeStableReleaseCurriculumEnv",
]
