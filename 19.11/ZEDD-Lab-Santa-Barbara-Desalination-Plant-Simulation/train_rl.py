#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rl.py — Train PPO agents on the Santa Barbara water supply environment.

Trains two agents:
  1. cost_only  — reward is purely -monthly_cost, no safety penalty
  2. safe       — reward includes a heavy penalty when months-of-supply < threshold

This lets us demonstrate the alignment/safety concern directly:
the cost_only agent learns to minimise cost but may risk water supply failure,
while the safe agent learns to balance cost against supply security.

Usage (from project root):
    python src/train_rl.py

Outputs written to result/rl/:
    agents/cost_only_agent/   — saved PPO model (stable-baselines3 format)
    agents/safe_agent/        — saved PPO model
    trajectories/cost_only_drought.npz   — episode trajectory on severe drought
    trajectories/safe_drought.npz        — episode trajectory on severe drought
    logs/cost_only/           — tensorboard logs
    logs/safe/                — tensorboard logs
"""

import os
import sys
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR      = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from water_env import WaterEnv

# ── Output directories ────────────────────────────────────────────────────────
RESULT_DIR   = os.path.join(PROJECT_ROOT, 'result', 'rl')
AGENT_DIR    = os.path.join(RESULT_DIR, 'agents')
TRAJ_DIR     = os.path.join(RESULT_DIR, 'trajectories')
LOG_DIR      = os.path.join(RESULT_DIR, 'logs')

for d in [AGENT_DIR, TRAJ_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Shared environment config ─────────────────────────────────────────────────
ENV_KWARGS_BASE = dict(
    case_number             = "basetariff_flexible/4mpd_36vessels",
    drought_type            = "pers87_sev0.83n_4",
    cost_curves_dir         = "cost_curves/new_data",
    data_dir                = "data",
    randomise_init_storage  = True,   # randomise init for training robustness
    cost_scale              = 1e5,
)

# Severe drought scenario used for final evaluation
SEVERE_DROUGHT   = "pers87_sev0.83n_4"
EVAL_SCENARIO    = 0   # fix scenario index for comparable eval

# ── Training config ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 500_000   # increase to 1_000_000 for better convergence
N_ENVS           = 4         # parallel environments
EVAL_FREQ        = 20_000    # evaluate every N steps
N_EVAL_EPISODES  = 5

# PPO hyperparameters — tuned for continuous control
PPO_KWARGS = dict(
    learning_rate   = 3e-4,
    n_steps         = 2048,
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,   # small entropy bonus to encourage exploration
    verbose         = 1,
)


def make_env(safety_penalty: float = 0.0, scenario_idx=None):
    """Factory function returning a monitored WaterEnv."""
    def _init():
        env = WaterEnv(
            **ENV_KWARGS_BASE,
            safety_penalty          = safety_penalty,
            safety_threshold_months = 3.0,
            scenario_idx            = scenario_idx,
        )
        return Monitor(env)
    return _init


def train_agent(name: str, safety_penalty: float) -> PPO:
    """
    Train a PPO agent and save it.

    Parameters
    ----------
    name : str
        Agent name — used for file paths and logging.
    safety_penalty : float
        Safety penalty magnitude. 0.0 = cost-only, >0 = safety-constrained.

    Returns
    -------
    Trained PPO model.
    """
    print(f"\n{'='*60}")
    print(f"Training agent: {name}")
    print(f"  Safety penalty: {safety_penalty}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs: {N_ENVS}")
    print(f"{'='*60}\n")

    # Training environments — randomised init storage, random scenario each episode
    train_env = make_vec_env(
        make_env(safety_penalty=safety_penalty),
        n_envs=N_ENVS,
        seed=42,
    )

    # Evaluation environment — fixed scenario for comparable metrics
    eval_env = Monitor(WaterEnv(
        **ENV_KWARGS_BASE,
        safety_penalty          = safety_penalty,
        safety_threshold_months = 3.0,
        scenario_idx            = EVAL_SCENARIO,
        randomise_init_storage  = False,
    ))

    agent_save_path = os.path.join(AGENT_DIR, name)
    log_path        = os.path.join(LOG_DIR, name)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = agent_save_path,
        log_path             = log_path,
        eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(100_000 // N_ENVS, 1),
        save_path   = agent_save_path,
        name_prefix = name,
        verbose     = 1,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log = log_path,
        **PPO_KWARGS,
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, checkpoint_callback],
        progress_bar    = True,
    )

    # Save final model
    final_path = os.path.join(agent_save_path, f"{name}_final")
    model.save(final_path)
    print(f"\n  Saved final model → {final_path}.zip")

    train_env.close()
    eval_env.close()

    return model


def run_eval_episode(model: PPO, safety_penalty: float, drought_type: str, scenario_idx: int):
    """
    Run a single deterministic episode and return the full trajectory.
    Used to compare agents on the same drought scenario.
    """
    env = WaterEnv(
        **ENV_KWARGS_BASE,
        drought_type            = drought_type,
        safety_penalty          = safety_penalty,
        safety_threshold_months = 3.0,
        scenario_idx            = scenario_idx,
        randomise_init_storage  = False,
    )

    obs, _ = env.reset(seed=0)
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    traj = env.get_trajectory()

    print(f"    Total cost:    ${info['episode_total_cost']:>15,.0f}")
    print(f"    Mean risk:     {info['episode_mean_risk']:>8.1f} months")
    print(f"    Min risk:      {info['episode_min_risk']:>8.1f} months")
    print(f"    Total deficit: {info['episode_total_deficit']:>8.1f} AF")

    return traj, info


def save_trajectory(traj: dict, info: dict, path: str):
    """Save trajectory arrays to a .npz file for later visualisation."""
    np.savez(
        path,
        sc            = traj["sc"],
        sgi           = traj["sgi"],
        sswp          = traj["sswp"],
        desal         = traj["desal"],
        cost          = traj["cost"],
        risk          = traj["risk"],
        deficit       = traj["deficit"],
        reward        = traj["reward"],
        desal_capacity = traj["desal_capacity"],
        demand        = traj["demand"],
        episode_total_cost    = info["episode_total_cost"],
        episode_mean_risk     = info["episode_mean_risk"],
        episode_min_risk      = info["episode_min_risk"],
        episode_total_deficit = info["episode_total_deficit"],
    )
    print(f"    Trajectory saved → {path}.npz")


if __name__ == "__main__":

    # ── 1. Train cost-only agent ───────────────────────────────────────────────
    cost_only_model = train_agent(
        name           = "cost_only_agent",
        safety_penalty = 0.0,
    )

    # ── 2. Train safety-constrained agent ─────────────────────────────────────
    safe_model = train_agent(
        name           = "safe_agent",
        safety_penalty = 5000.0,
    )

    # ── 3. Evaluate both on severe drought ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluating both agents on severe drought: {SEVERE_DROUGHT}")
    print(f"{'='*60}")

    print("\nCost-only agent:")
    cost_traj, cost_info = run_eval_episode(
        cost_only_model, 0.0, SEVERE_DROUGHT, EVAL_SCENARIO
    )
    save_trajectory(
        cost_traj, cost_info,
        os.path.join(TRAJ_DIR, "cost_only_drought")
    )

    print("\nSafe agent:")
    safe_traj, safe_info = run_eval_episode(
        safe_model, 5000.0, SEVERE_DROUGHT, EVAL_SCENARIO
    )
    save_trajectory(
        safe_traj, safe_info,
        os.path.join(TRAJ_DIR, "safe_drought")
    )

    # ── 4. Summary comparison ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — AGENT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Cost-only':>15} {'Safe':>15}")
    print(f"{'-'*60}")
    print(f"{'Total cost ($)':<30} {cost_info['episode_total_cost']:>15,.0f} {safe_info['episode_total_cost']:>15,.0f}")
    print(f"{'Mean risk (months)':<30} {cost_info['episode_mean_risk']:>15.1f} {safe_info['episode_mean_risk']:>15.1f}")
    print(f"{'Min risk (months)':<30} {cost_info['episode_min_risk']:>15.1f} {safe_info['episode_min_risk']:>15.1f}")
    print(f"{'Total deficit (AF)':<30} {cost_info['episode_total_deficit']:>15.1f} {safe_info['episode_total_deficit']:>15.1f}")
    print(f"{'='*60}")
    print("\nOutputs saved to result/rl/")
    print("  agents/cost_only_agent/  — trained PPO model")
    print("  agents/safe_agent/       — trained PPO model")
    print("  trajectories/            — .npz files for visualisation")
    print("  logs/                    — tensorboard logs")
    print("\nNext step: python src/visualize_rl.py")
