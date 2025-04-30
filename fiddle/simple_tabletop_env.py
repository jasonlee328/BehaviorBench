import argparse
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.wrappers.record import RecordEpisode

import objaverse


ITEMS_YCB_LIST = [
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
]

ITEMS_OBJAVERSE_LIST = [
    # "8476c4170df24cf5bbe6967222d1a42d",
    "8ff7f1f2465347cd8b80c9b206c2781e",
    # "c786b97d08b94d02a1fa3b87d2e86cf1",
    # "139331da744542009f146018fd0e05f4",
    # "be2c02614d774f9da672dfdc44015219",
]


@register_env("SimpleTableTop-v1", max_episode_steps=200)
class SimpleTableTopEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        for item_id in ITEMS_YCB_LIST:
            builder = self._load_model_ycb(item_id)
            x, y = np.random.uniform(-0.2, 0.2, size=2)
            builder.initial_pose = sapien.Pose(p=[x, y, 0.1])
            builder.build(name=f"ycb-model-{item_id}")

        # Load models from Objaverse
        objs_cache = objaverse.load_objects(ITEMS_OBJAVERSE_LIST)
        for _, obj_path in objs_cache.items():
            builder = self._load_model_objaverse(obj_path)
            x, y = np.random.uniform(-0.2, 0.2, size=2)
            builder.initial_pose = sapien.Pose(p=[x, y, 0.7])
            builder.build(name=f"objaverse-model-{obj_path}")

    def _load_model_ycb(self, model_id) -> ActorBuilder:
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        return builder

    def _load_model_objaverse(self, model_path) -> ActorBuilder:
        scale = 0.1
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(filename=model_path, scale=(scale,) * 3)
        builder.add_multiple_convex_collisions_from_file(
            filename=model_path,
            decomposition="coacd",
            density=1000,
            material=None,
            scale=(scale,) * 3,
        )
        return builder

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self) -> dict:
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1


def main() -> int:
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        action="store_true",
        help="Flag used to record a video",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Flag used to run the environment in headless mode",
    )

    args = parser.parse_args()

    env = gym.make(
        "SimpleTableTop-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human" if not args.headless else "rgb_array",
    )

    if args.record:
        env = RecordEpisode(env, output_dir="recordings")

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()  # a display is required to render
    env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
