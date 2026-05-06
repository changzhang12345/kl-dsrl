import math
import os
import random
import sys
import warnings

warnings.filterwarnings("ignore")

import d4rl
import d4rl.gym_mujoco
import gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dsrl.kl_dsrl import KLDSRL

sys.path.append("./dppo")

from env_utils import ActionChunkWrapper, ObservationWrapperGym, ObservationWrapperRobomimic, make_robomimic_env
from utils import load_base_policy

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
	config_path=os.path.join(base_path, "cfg/robomimic"),
	config_name="kl_dsrl_square.yaml",
	version_base=None,
)
def main(cfg: OmegaConf):
	OmegaConf.resolve(cfg)

	checkpoint_path = cfg.get("checkpoint_path", None)
	if checkpoint_path is None:
		raise ValueError("Pass checkpoint_path=/path/to/final.zip")

	n_eval_episodes = int(cfg.get("n_eval_episodes", cfg.num_evals))
	deterministic = bool(cfg.get("eval_deterministic", False))

	random.seed(cfg.seed)
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)

	def make_env():
		if cfg.env_name in ["halfcheetah-medium-v2", "hopper-medium-v2", "walker2d-medium-v2"]:
			env = gym.make(cfg.env_name)
			env = ObservationWrapperGym(env, cfg.normalization_path)
		elif cfg.env_name in ["lift", "can", "square", "transport"]:
			env = make_robomimic_env(
				env=cfg.env_name,
				normalization_path=cfg.normalization_path,
				low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
				dppo_path=cfg.dppo_path,
				reward_shaping=cfg.env.get("reward_shaping", False),
			)
			env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
		else:
			raise ValueError(f"Unsupported env_name: {cfg.env_name}")
		return ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)

	base_policy = load_base_policy(cfg)
	env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
	env.seed(cfg.seed + 1)

	model = KLDSRL.load(
		checkpoint_path,
		env=env,
		device=cfg.device,
		diffusion_policy=base_policy,
		diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
	)
	model.diffusion_policy = base_policy
	model.diffusion_act_chunk = cfg.act_steps
	model.diffusion_act_dim = cfg.action_dim

	max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
	successes, returns = [], []

	for ep in range(n_eval_episodes):
		obs = env.reset()
		ep_return = 0.0
		success = False
		for _ in range(max_steps):
			action, _ = model.predict_diffused(obs, deterministic=deterministic)
			obs, reward, done, infos = env.step(action)
			ep_return += float(np.sum(reward))
			success_infos = [info.get("is_success") for info in infos if "is_success" in info]
			if success_infos:
				success = success or bool(np.any(success_infos))
			else:
				success = success or bool(np.any(reward > -cfg.env.reward_offset))
			if bool(np.any(done)):
				break
		successes.append(float(success))
		returns.append(ep_return)
		print(f"episode={ep} success={int(success)} return={ep_return:.3f}")

	print(f"success_rate={np.mean(successes):.3f} mean_return={np.mean(returns):.3f}")
	env.close()


if __name__ == "__main__":
	main()
