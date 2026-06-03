from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # Algorithm / environment arguments
    env_id: str = "PickCube-v1"
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    num_envs: int = 512
    num_eval_envs: int = 8
    partial_reset: bool = True
    eval_partial_reset: bool = False
    num_steps: int = 50
    num_eval_steps: int = 50
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    anneal_lr: bool = False
    gamma: float = 0.8
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None
    finite_horizon_gae: bool = False

    # Filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def finalize_args(args: Args) -> Args:
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def make_run_name(args: Args, filename: str = "ppo_rgb.py") -> str:
    if args.exp_name is None:
        exp_name = os.path.basename(filename)
        if exp_name.endswith(".py"):
            exp_name = exp_name[:-3]
        return f"{args.env_id}__{exp_name}__{args.seed}__{int(time.time())}"
    return args.exp_name


def seed_everything(args: Args) -> torch.device:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    return torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic: bool = False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class Logger:
    def __init__(self, log_wandb: bool = False, tensorboard: Optional[SummaryWriter] = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def make_envs(args: Args, run_name: str):
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate and args.checkpoint:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")

        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )

        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    return envs, eval_envs, env_kwargs, max_episode_steps


def make_logger(args: Args, run_name: str, env_kwargs: dict, max_episode_steps: int) -> Optional[Logger]:
    if args.evaluate:
        print("Running evaluation")
        return None

    print("Running training")
    if args.track:
        import wandb
        config = vars(args)
        config["env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_envs,
            env_id=args.env_id,
            reward_mode="normalized_dense",
            env_horizon=max_episode_steps,
            partial_reset=args.partial_reset,
        )
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            reward_mode="normalized_dense",
            env_horizon=max_episode_steps,
            partial_reset=False,
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group="PPO",
            tags=["ppo", "walltime_efficient"],
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    return Logger(log_wandb=args.track, tensorboard=writer)


def make_action_clipper(envs, device):
    low = torch.from_numpy(envs.single_action_space.low).to(device)
    high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), low, high)

    return clip_action


def evaluate_policy(agent: Agent, eval_envs, args: Args, device, global_step: int, logger: Optional[Logger], start_time: float):
    print("Evaluating")
    eval_obs, _ = eval_envs.reset()
    eval_metrics = defaultdict(list)
    num_episodes = 0

    agent.eval()
    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                agent.get_action(eval_obs, deterministic=True)
            )
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                num_episodes += int(mask.sum().item() if hasattr(mask.sum(), "item") else mask.sum())
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v)

    print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")

    out = {
        "num_episodes": float(num_episodes),
        "wall_time_min": (time.time() - start_time) / 60.0,
        "global_step": float(global_step),
    }
    for k, v in eval_metrics.items():
        mean = torch.stack(v).float().mean()
        val = float(mean.detach().cpu().item())
        out[k] = val
        if logger is not None:
            logger.add_scalar(f"eval/{k}", mean, global_step)
        print(f"eval_{k}_mean={mean}")

    # Normalize common ManiSkill naming variants into stable summary keys.
    success = None
    for key in ("success", "success_once", "is_success"):
        if key in out:
            success = out[key]
            break
    if success is not None:
        out["success_rate"] = float(success)
    if "return" in out:
        out["return_mean"] = float(out["return"])
    elif "reward" in out:
        out["return_mean"] = float(out["reward"])
    elif "r" in out:
        out["return_mean"] = float(out["r"])
    return out


def summarize_evals(eval_history, start_time: float, global_step: int) -> dict:
    total_seconds = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    sps = int(global_step / max(total_seconds, 1e-9))

    successes = [(e["wall_time_min"], e["success_rate"]) for e in eval_history if "success_rate" in e]
    returns = [(e["wall_time_min"], e["return_mean"]) for e in eval_history if "return_mean" in e]

    def first_time_at(threshold: float) -> float:
        for t, s in successes:
            if s >= threshold:
                return t
        return -1.0

    if successes:
        final_success = successes[-1][1]
        best_success = max(s for _, s in successes)
        if len(successes) >= 2:
            area = 0.0
            for (t0, s0), (t1, s1) in zip(successes[:-1], successes[1:]):
                area += (t1 - t0) * (s0 + s1) / 2.0
            auc_success_per_min = area / max(successes[-1][0] - successes[0][0], 1e-9)
        else:
            auc_success_per_min = final_success
    else:
        final_success = 0.0
        best_success = 0.0
        auc_success_per_min = 0.0

    final_return = returns[-1][1] if returns else 0.0

    return {
        "final_eval_success_rate": float(final_success),
        "final_eval_return_mean": float(final_return),
        "best_eval_success_rate": float(best_success),
        "auc_eval_success_per_min": float(auc_success_per_min),
        "time_to_50_success_min": float(first_time_at(0.50)),
        "time_to_80_success_min": float(first_time_at(0.80)),
        "total_seconds": float(total_seconds),
        "SPS": int(sps),
        "peak_vram_mb": float(peak_vram_mb),
        "global_step": int(global_step),
    }


def print_final_summary(summary: dict) -> None:
    print("---")
    print(f"final_eval_success_rate: {summary['final_eval_success_rate']:.6f}")
    print(f"final_eval_return_mean: {summary['final_eval_return_mean']:.6f}")
    print(f"best_eval_success_rate: {summary['best_eval_success_rate']:.6f}")
    print(f"auc_eval_success_per_min: {summary['auc_eval_success_per_min']:.6f}")
    print(f"time_to_50_success_min: {summary['time_to_50_success_min']:.1f}")
    print(f"time_to_80_success_min: {summary['time_to_80_success_min']:.1f}")
    print(f"total_seconds: {summary['total_seconds']:.1f}")
    print(f"SPS: {summary['SPS']}")
    print(f"peak_vram_mb: {summary['peak_vram_mb']:.1f}")
    print(f"global_step: {summary['global_step']}")
