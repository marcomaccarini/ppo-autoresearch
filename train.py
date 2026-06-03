import csv
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from prepare import (
    Agent,
    Args,
    evaluate_policy,
    finalize_args,
    make_action_clipper,
    make_envs,
    make_logger,
    make_run_name,
    print_final_summary,
    seed_everything,
    summarize_evals,
)


SEGMENT_STEPS = 1_000_000

_SEGMENT_COLS = [
    "segment", "global_step", "commit", "eval_success_rate", "eval_return_mean",
    "auc_success_per_mstep", "SPS", "approx_kl", "clipfrac", "entropy",
    "value_loss", "policy_loss", "explained_variance", "wall_time_sec",
    "status", "description",
]


def _append_segment_metrics(run_name, segment, eval_metrics, diag, eval_history, start_time):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = ""

    successes = [(e["global_step"], e.get("success_rate", 0.0))
                 for e in eval_history if "success_rate" in e]
    if len(successes) >= 2:
        xs, ys = [s for s, _ in successes], [r for _, r in successes]
        area = sum((xs[k+1]-xs[k])*(ys[k]+ys[k+1])/2 for k in range(len(xs)-1))
        auc = area / max(xs[-1] - xs[0], 1)
    else:
        auc = successes[-1][1] if successes else 0.0

    wall_time_sec = time.time() - start_time
    gs = int(eval_metrics.get("global_step", 0))
    sps = int(gs / max(wall_time_sec, 1e-9))

    def _fmt(v):
        return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    row = {
        "segment": segment,
        "global_step": gs,
        "commit": commit,
        "eval_success_rate": _fmt(eval_metrics.get("success_rate")),
        "eval_return_mean": _fmt(eval_metrics.get("return_mean")),
        "auc_success_per_mstep": round(auc, 6),
        "SPS": sps,
        "approx_kl": _fmt(diag.get("approx_kl")),
        "clipfrac": _fmt(diag.get("clipfrac")),
        "entropy": _fmt(diag.get("entropy")),
        "value_loss": _fmt(diag.get("value_loss")),
        "policy_loss": _fmt(diag.get("policy_loss")),
        "explained_variance": _fmt(diag.get("explained_variance")),
        "wall_time_sec": round(wall_time_sec, 1),
        "status": "segment_done",
        "description": "",
    }

    tsv_path = f"runs/{run_name}/segment_metrics.tsv"
    jsonl_path = f"runs/{run_name}/segment_metrics.jsonl"
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SEGMENT_COLS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    args = finalize_args(tyro.cli(Args))
    run_name = make_run_name(args, filename="ppo_rgb.py")
    device = seed_everything(args)

    envs, eval_envs, env_kwargs, max_episode_steps = make_envs(args, run_name)
    logger = make_logger(args, run_name, env_kwargs, max_episode_steps)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    print("####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print("####")

    clip_action = make_action_clipper(envs, device)

    ckpt_dir = f"runs/{run_name}/checkpoints"
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")

    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt["global_step"]
        print(f"Resumed from {latest_ckpt}: global_step={global_step}")
    elif args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    segment_end_step = min(global_step + SEGMENT_STEPS, args.total_timesteps)
    print(f"Segment: global_step={global_step} -> {segment_end_step}")

    eval_history = []
    _eval_count = 0
    tsv_path = f"runs/{run_name}/segment_metrics.tsv"
    if os.path.exists(tsv_path):
        with open(tsv_path) as f:
            _eval_count = max(0, sum(1 for _ in f) - 1)
    _last_diag: dict = {}

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        if iteration % args.eval_freq == 1:
            metrics = evaluate_policy(agent, eval_envs, args, device, global_step, logger, start_time)
            eval_history.append(metrics)
            if args.evaluate:
                break
            _eval_count += 1
            _append_segment_metrics(run_name, _eval_count, metrics, _last_diag, eval_history, start_time)

        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Anneal LR over the full 20M-step run to stabilise late training
        lr_frac = max(1.0 - global_step / args.total_timesteps, 0.0)
        optimizer.param_groups[0]["lr"] = lr_frac * args.learning_rate

        rollout_time = time.time()
        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos and logger is not None:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                        infos["final_observation"][done_mask]
                    ).view(-1)

        rollout_time = time.time() - rollout_time

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                real_next_values = next_not_done * nextvalues + final_values[t]

                if args.finite_horizon_gae:
                    if t == args.num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()

        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            early_stop = False
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                eff_target_kl = 0.02  # tighter than args.target_kl to curb rising clipfrac
                if approx_kl > eff_target_kl:
                    early_stop = True
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                ent_coef = 0.01  # counter fast entropy collapse seen in seg 1 (6.4->4.6 nats)
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    agent.actor_logstd.clamp_(min=-2.0)  # floor entropy ~-4 for 8-dim action

            if early_stop:
                break

        update_time = time.time() - update_time

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        sps = int(global_step / (time.time() - start_time))

        _last_diag = {
            "approx_kl": float(approx_kl.item()),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
            "entropy": float(entropy_loss.item()),
            "value_loss": float(v_loss.item()),
            "policy_loss": float(pg_loss.item()),
            "explained_variance": float(explained_var),
        }

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs) if clipfracs else 0.0, global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

        print("SPS:", sps)

        if global_step >= segment_end_step:
            break

    if not args.evaluate:
        # Final deterministic eval so the summary always reflects the final policy.
        metrics = evaluate_policy(agent, eval_envs, args, device, global_step, logger, start_time)
        eval_history.append(metrics)
        _eval_count += 1
        _append_segment_metrics(run_name, _eval_count, metrics, _last_diag, eval_history, start_time)

        if args.save_model:
            os.makedirs(ckpt_dir, exist_ok=True)
            if os.path.exists(latest_ckpt):
                shutil.copy2(latest_ckpt, os.path.join(ckpt_dir, "before_segment.pt"))
            torch.save({"global_step": global_step, "agent": agent.state_dict(), "optimizer": optimizer.state_dict()}, latest_ckpt)
            print(f"Segment checkpoint saved: {latest_ckpt} (global_step={global_step})")
            if global_step >= args.total_timesteps:
                model_path = f"runs/{run_name}/final_ckpt.pt"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()

    summary = summarize_evals(eval_history, start_time, global_step)
    print_final_summary(summary)

    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main()
