from pathlib import Path

import gymnasium as gym

from omegaconf import DictConfig, OmegaConf

from behavior_bench import CONFIG_DIR
from behavior_bench.core.environment import BaseTableTopEnv


def main() -> int:
    env_config = OmegaConf.load(CONFIG_DIR / "simple_tabletop_v1.yaml")

    env = gym.make(
        "BaseTableTop-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
        env_config=env_config,
    )

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()

    return 0




if __name__ == "__main__":
    raise SystemExit(main())



