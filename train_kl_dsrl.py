import os
import warnings
warnings.filterwarnings("ignore")
import math
import torch
import random
import wandb
import numpy as np
import hydra
from omegaconf import OmegaConf
import gym, d4rl
import d4rl.gym_mujoco
import sys
sys.path.append('./dppo')

from stable_baselines3.dsrl.kl_dsrl import KLDSRL
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from env_utils import DiffusionPolicyEnvWrapper, ObservationWrapperRobomimic, ObservationWrapperGym, ActionChunkWrapper, make_robomimic_env
from utils import load_base_policy, load_offline_data, collect_rollouts, collect_policy_rollouts, LoggingCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

base_path = os.path.dirname(os.path.abspath(__file__))


@hydra.main(
    config_path=os.path.join(base_path, "cfg/robomimic"), config_name="kl_dsrl_can.yaml", version_base=None
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.name,
            group=cfg.wandb.group,
            monitor_gym=True,
            save_code=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    MAX_STEPS = int(cfg.env.max_episode_steps / cfg.act_steps)

    num_env = cfg.env.n_envs
    def make_env():
        if cfg.env_name in ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']:
            env = gym.make(cfg.env_name)
            env = ObservationWrapperGym(env, cfg.normalization_path)
        elif cfg.env_name in ['lift', 'can', 'square', 'transport']:
            env = make_robomimic_env(
                env=cfg.env_name,
                normalization_path=cfg.normalization_path,
                low_dim_keys=cfg.env.wrappers.robomimic_lowdim.low_dim_keys,
                dppo_path=cfg.dppo_path,
                reward_shaping=cfg.env.get("reward_shaping", False),
            )
            env = ObservationWrapperRobomimic(env, reward_offset=cfg.env.reward_offset)
        env = ActionChunkWrapper(env, cfg, max_episode_steps=cfg.env.max_episode_steps)
        return env

    base_policy = load_base_policy(cfg)
    env = make_vec_env(make_env, n_envs=num_env, vec_env_cls=SubprocVecEnv)
    env.seed(cfg.seed + 1)

    post_linear_modules = None
    if cfg.train.use_layer_norm:
        post_linear_modules = [torch.nn.LayerNorm]

    net_arch = []
    for _ in range(cfg.train.num_layers):
        net_arch.append(cfg.train.layer_size)
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        log_std_init=0.0,
        post_linear_modules=post_linear_modules,
        n_critics=cfg.train.n_critics,
    )

    resume_path = cfg.get("resume_path", None)
    if resume_path:
        model = KLDSRL.load(
            resume_path,
            env=env,
            device=cfg.device,
            tensorboard_log=cfg.logdir,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
        )
        model.diffusion_policy = base_policy
        model.diffusion_act_chunk = cfg.act_steps
        model.diffusion_act_dim = cfg.action_dim
        assert model.kl_coef == cfg.train.kl_coef, (
            f"checkpoint kl_coef={model.kl_coef} != train.kl_coef={cfg.train.kl_coef}"
        )
        print(f"Resumed from {resume_path} at num_timesteps={model.num_timesteps}")
    else:
        model = KLDSRL(
            "MlpPolicy",
            env,
            learning_rate=cfg.train.actor_lr,
            buffer_size=cfg.train.buffer_size,
            learning_starts=1,
            batch_size=cfg.train.batch_size,
            tau=cfg.train.tau,
            gamma=cfg.train.discount,
            train_freq=cfg.train.train_freq,
            gradient_steps=cfg.train.utd,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef="auto" if cfg.train.ent_coef == -1 else cfg.train.ent_coef,
            target_update_interval=1,
            target_entropy="auto" if cfg.train.target_ent == -1 else cfg.train.target_ent,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=cfg.logdir,
            verbose=1,
            policy_kwargs=policy_kwargs,
            diffusion_policy=base_policy,
            diffusion_act_dim=(cfg.act_steps, cfg.action_dim),
            noise_critic_grad_steps=cfg.train.noise_critic_grad_steps,
            critic_backup_combine_type=cfg.train.critic_backup_combine_type,
            kl_coef=cfg.train.kl_coef,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_model_interval,
        save_path=cfg.logdir + '/checkpoint/',
        name_prefix='ft_policy',
        save_replay_buffer=cfg.save_replay_buffer,
        save_vecnormalize=True,
    )

    num_env_eval = cfg.env.n_eval_envs
    eval_env = make_vec_env(make_env, n_envs=num_env_eval, vec_env_cls=SubprocVecEnv)
    eval_env.seed(cfg.seed + num_env + 1)

    logging_callback = LoggingCallback(
        action_chunk=cfg.act_steps,
        eval_episodes=int(cfg.num_evals / num_env_eval),
        log_freq=MAX_STEPS,
        use_wandb=cfg.use_wandb,
        eval_env=eval_env,
        eval_freq=cfg.eval_interval,
        num_train_env=num_env,
        num_eval_env=num_env_eval,
        rew_offset=cfg.env.reward_offset,
        algorithm=cfg.algorithm,
        max_steps=MAX_STEPS,
        deterministic_eval=cfg.deterministic_eval,
    )

    logging_callback.evaluate(model, deterministic=False)
    if cfg.deterministic_eval:
        logging_callback.evaluate(model, deterministic=True)
    logging_callback.log_count += 1

    if resume_path:
        if cfg.get("resume_replay_buffer", None):
            model.load_replay_buffer(cfg.resume_replay_buffer)
        else:
            # No saved buffer: refill it with the resumed policy before updating.
            warmup_steps = cfg.get("resume_warmup_steps", cfg.train.init_rollout_steps)
            collect_policy_rollouts(model, env, warmup_steps)
        logging_callback.set_timesteps(model.num_timesteps * cfg.act_steps)
    else:
        if cfg.load_offline_data:
            load_offline_data(model, cfg.offline_data_path, num_env)
        if cfg.train.init_rollout_steps > 0:
            collect_rollouts(model, env, cfg.train.init_rollout_steps, base_policy, cfg)
            logging_callback.set_timesteps(cfg.train.init_rollout_steps * num_env)

    callbacks = [checkpoint_callback, logging_callback]
    # When resuming, total_timesteps is the target overall step count (e.g. 500000 continues 250000 -> 500000).
    model.learn(
        total_timesteps=cfg.total_timesteps - model.num_timesteps if resume_path else cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=not resume_path,
    )

    if len(cfg.name) > 0:
        model.save(cfg.logdir + "/checkpoint/final")
        if cfg.get("save_final_replay_buffer", False):
            model.save_replay_buffer(cfg.logdir + "/checkpoint/final_replay_buffer")

    env.close()
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
