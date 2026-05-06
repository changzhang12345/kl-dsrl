import csv
import math
import os
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import d4rl
import d4rl.gym_mujoco
import gym
import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dsrl.kl_dsrl import KLDSRL

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("./dppo")

from env_utils import ActionChunkWrapper, ObservationWrapperGym, ObservationWrapperRobomimic, make_robomimic_env
from utils import load_base_policy


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RUNS = {
    "0": "logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta0_seed1",
    "0.1": "logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta0p1_seed1",
    "0.5": "logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta0p5_seed1",
    "1": "logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta1_seed1",
    "2": "logs/robomimic-kl-dsrl/robomimic_square_kl_dsrl_beta2_seed1",
}

DEFAULT_DENSE_RUNS = {
    "0": "logs/robomimic-kl-dsrl-dense/robomimic_square_kl_dsrl_dense_beta0_seed1",
    "0.1": "logs/robomimic-kl-dsrl-dense/robomimic_square_kl_dsrl_dense_beta0p1_seed1",
    "0.5": "logs/robomimic-kl-dsrl-dense/robomimic_square_kl_dsrl_dense_beta0p5_seed1",
    "1": "logs/robomimic-kl-dsrl-dense/robomimic_square_kl_dsrl_dense_beta1_seed1",
    "2": "logs/robomimic-kl-dsrl-dense/robomimic_square_kl_dsrl_dense_beta2_seed1",
}

STEP_RE = re.compile(r"ft_policy_(\d+)_steps\.zip$")


def make_eval_env_fn(cfg):
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

    return make_env


def find_checkpoints(run_dir, final_step):
    checkpoints = []
    for path in sorted(Path(run_dir).glob("*/checkpoint/*.zip")):
        match = STEP_RE.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
        elif path.name == "final.zip":
            checkpoints.append((int(final_step), path))
    checkpoints.sort(key=lambda item: (item[0], item[1].name == "final.zip"))
    return checkpoints


def evaluate_checkpoint(cfg, env, base_policy, checkpoint_path, n_eval_episodes, deterministic):
    model = KLDSRL.load(
        str(checkpoint_path),
        env=env,
        device=cfg.device,
        diffusion_policy=base_policy,
        diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
    )
    model.diffusion_policy = base_policy
    model.diffusion_act_chunk = cfg.act_steps
    model.diffusion_act_dim = cfg.action_dim

    max_steps = int(cfg.env.max_episode_steps / cfg.act_steps)
    successes, returns, lengths = [], [], []

    with torch.no_grad():
        for ep in range(n_eval_episodes):
            obs = env.reset()
            ep_return = 0.0
            success = False
            ep_len = 0
            for _ in range(max_steps):
                action, _ = model.predict_diffused(obs, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                ep_return += float(np.sum(reward))
                ep_len += int(cfg.act_steps)
                success_infos = [info.get("is_success") for info in infos if "is_success" in info]
                if success_infos:
                    success = success or bool(np.any(success_infos))
                else:
                    success = success or bool(np.any(reward > -cfg.env.reward_offset))
                if bool(np.any(done)):
                    break
            successes.append(float(success))
            returns.append(ep_return)
            lengths.append(ep_len)
            print(f"  episode={ep} success={int(success)} return={ep_return:.3f} len={ep_len}")

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_episode_length": float(np.mean(lengths)),
    }


def write_csv(rows, csv_path):
    fieldnames = [
        "beta",
        "checkpoint_step",
        "total_timesteps",
        "checkpoint_name",
        "checkpoint_path",
        "success_rate",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_episode_length",
        "num_episodes",
        "deterministic",
        "seed",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, metric, ylabel, output_path):
    plt.figure(figsize=(8, 5))
    betas = sorted({row["beta"] for row in rows}, key=float)
    for beta in betas:
        beta_rows = [row for row in rows if row["beta"] == beta]
        beta_rows.sort(key=lambda row: row["total_timesteps"])
        x = [row["total_timesteps"] for row in beta_rows]
        y = [row[metric] for row in beta_rows]
        plt.plot(x, y, marker="o", linewidth=2, label=f"beta={beta}")
    plt.xlabel("Total timesteps")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"),
    config_name="kl_dsrl_square.yaml",
    version_base=None,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n_eval_episodes = int(cfg.get("n_eval_episodes", cfg.num_evals))
    deterministic = bool(cfg.get("eval_deterministic", False))
    is_dense = bool(cfg.env.get("reward_shaping", False))
    default_output_dir = "eval/kl_dsrl_dense_sweep" if is_dense else "eval/kl_dsrl_sweep"
    output_dir = Path(to_absolute_path(str(cfg.get("eval_output_dir", default_output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)

    timestep_multiplier = int(cfg.get("plot_timestep_multiplier", cfg.env.n_envs))
    run_map = DEFAULT_DENSE_RUNS if is_dense else DEFAULT_RUNS

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_eval_env_fn(cfg), n_envs=1, vec_env_cls=DummyVecEnv)
    env.seed(cfg.seed + 1)

    rows = []
    for beta, run_dir in run_map.items():
        abs_run_dir = Path(to_absolute_path(run_dir))
        checkpoints = find_checkpoints(abs_run_dir, final_step=cfg.total_timesteps)
        if not checkpoints:
            print(f"[WARN] no checkpoints found for beta={beta}: {abs_run_dir}")
            continue

        for checkpoint_step, checkpoint_path in checkpoints:
            total_timesteps = checkpoint_step * timestep_multiplier
            print(f"Evaluating beta={beta} step={checkpoint_step} total_timesteps={total_timesteps}")
            print(f"  checkpoint={checkpoint_path}")
            metrics = evaluate_checkpoint(
                cfg=cfg,
                env=env,
                base_policy=base_policy,
                checkpoint_path=checkpoint_path,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
            )
            row = {
                "beta": beta,
                "checkpoint_step": checkpoint_step,
                "total_timesteps": total_timesteps,
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "num_episodes": n_eval_episodes,
                "deterministic": deterministic,
                "seed": cfg.seed,
            }
            row.update(metrics)
            rows.append(row)

            write_csv(rows, output_dir / "kl_dsrl_checkpoint_eval.csv")

    env.close()

    if rows:
        plot_metric(rows, "success_rate", "Success rate", output_dir / "success_rate_by_beta.png")
        plot_metric(rows, "mean_return", "Mean return", output_dir / "mean_return_by_beta.png")
        print(f"Saved CSV and plots to {output_dir}")
    else:
        print("No evaluation rows were produced.")


if __name__ == "__main__":
    main()
