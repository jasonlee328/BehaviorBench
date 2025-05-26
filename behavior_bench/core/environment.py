from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch

from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R

from mani_skill.agents.robots import Panda, XArm6Robotiq, SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("BaseTableTop-v1", max_episode_steps=2000, asset_download_ids=["ycb"])
class BaseTableTopEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "xarm6_robotiq",
        "so100",
    ]

    agent: Union[Panda, XArm6Robotiq, SO100]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        env_config: Optional[DictConfig] = None,
        **kwargs,
    ):
        self._count_actors = 0
        self._actors: List[Actor] = []
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._robot_init_p = [-0.615, 0, 0]
        self._robot_init_q = np.array([1, 0, 0, 0], dtype=np.float64)
        self.env_config = env_config or {}
        if env_config is not None and "robot" in env_config:
            self._robot_init_p = env_config["robot"].get("position", [-0.615, 0, 0])
            robot_init_euler = env_config["robot"].get("euler", [0, 0, 0])
            self._robot_init_q = R.from_euler('xyz', robot_init_euler, degrees=True).as_quat(scalar_first=True)
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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1

    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = None, build_separate: bool = False):
        super()._load_agent(options, sapien.Pose(p=self._robot_init_p, q=self._robot_init_q))

    def _load_scene(self, options: dict):
        self.table_scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene_builder.build()

        self.origin = actors.build_sphere(
            self.scene,
            radius=0.025,
            color=[0, 1, 0, 1],
            name="origin",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        if self.env_config is not None and "objects" in self.env_config:
            objects = self.env_config.get("objects", [])
            for obj in objects:
                obj_type = obj.get("type", "primitive")
                obj_id = obj.get("id", "sphere")
                obj_position = obj.get("position", [0, 0, 0])
                obj_euler = obj.get("euler", [0, 0, 0])

                actor: Optional[Actor] = None
                if obj_type == "primitive":
                    actor = self._load_primitive(
                        obj_id, obj_position, obj_euler
                    )
                elif obj_type == "ycb":
                    actor = self._load_ycb(
                        obj_id, obj_position, obj_euler
                    )

                if actor:
                    self._actors.append(actor)


    def _load_primitive(
        self,
        type: str = "sphere",
        pos: List[float] = [0, 0, 0],
        euler: List[float] = [0, 0, 0]
    ) -> Optional[Actor]:
        actor: Optional[Actor] = None
        quat = R.from_euler('xyz', euler, degrees=True).as_quat(scalar_first=True)
        if type == "sphere":
            actor = actors.build_sphere(
                self.scene,
                radius=0.025,
                color=[1, 0, 0, 1],
                name=f"sphere_{self._count_actors}",
                initial_pose=sapien.Pose(p=pos, q=quat),
            )
            self._count_actors += 1
        elif type == "box":
            actor = actors.build_box(
                self.scene,
                half_sizes=[0.025, 0.025, 0.025],
                color=[0, 1, 0, 1],
                name=f"box_{self._count_actors}",
                initial_pose=sapien.Pose(p=pos, q=quat),
            )
        elif type == "cylinder":
            actor = actors.build_cylinder(
                self.scene,
                radius=0.025,
                half_length=0.025,
                color=[0, 0, 1, 1],
                name=f"cylinder_{self._count_actors}",
                initial_pose=sapien.Pose(p=pos, q=quat),
            )
        return actor

    def _load_ycb(
        self,
        model_id: str,
        position: List[float],
        euler: List[float],
    ) -> Actor:
        quat = R.from_euler('xyz', euler, degrees=True).as_quat(scalar_first=True)
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
        builder.initial_pose = sapien.Pose(p=position, q=quat)
        return builder.build(name=f"ycb-model-{model_id}-{self._count_actors}")
