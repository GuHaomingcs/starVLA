"""RoboDojo released checkpoint schema: interleaved arm/gripper, q99 normalization.

Adapted from Cuzyoung/XPolicyLab's StarVLA RoboDojo data registration.
This describes the released training layout, not a new training reproduction.
"""

from __future__ import annotations


from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboDojoDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.left_joints": 6,
        "state.left_gripper": 1,
        "state.right_joints": 6,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.left_gripper": 1,
        "action.right_joints": 6,
        "action.right_gripper": 1,
    }
    modality_key_ranges = {
        "state": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
        "action": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.left_joints": "q99",
                        "state.left_gripper": "q99",
                        "state.right_joints": "q99",
                        "state.right_gripper": "q99",
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.left_joints": "q99",
                        "action.left_gripper": "q99",
                        "action.right_joints": "q99",
                        "action.right_gripper": "q99",
                    },
                ),
            ]
        )


class RoboDojoArxX5H50Q99DataConfig(RoboDojoDataConfig):
    """RoboDojo ARX X5 runtime schema for 50-step action chunks."""

    action_indices = list(range(50))


ROBOT_TYPE_CONFIG_MAP = {"robodojo_arx_x5_h50_q99": RoboDojoArxX5H50Q99DataConfig()}
DATASET_NAMED_MIXTURES = {"robodojo_v21_all_h50_q99": [("RoboDojo_lerobot_v21_video", 1.0, "robodojo_arx_x5_h50_q99")]}
