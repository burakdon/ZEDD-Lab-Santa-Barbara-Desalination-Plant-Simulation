#!/usr/bin/env python3
"""
diagnose_env.py — Quick diagnostic to inspect reward signal and agent behavior.
Run from project root: python3 src/diagnose_env.py
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

from water_env import WaterEnv

env = WaterEnv(
    case_number            = "basetariff_flexible/4mpd_36vessels",
    drought_type           = "pers87_sev0.83n_4",
    cost_curves_dir        = "cost_curves",
    data_dir               = "data",
    scenario_idx           = 0,
    randomise_init_storage = False,
    safety_penalty         = 50000.0,
    cost_scale             = 1e5,
)

obs, _ = env.reset(seed=0)
print(f"Initial obs:       {obs}")
print(f"Desal capacity:    {env.desal_capacity:.1f} AF/month")
print(f"Mean demand:       {np.mean(env.demand):.1f} AF/month")
print(f"Sustainable yield: {env.sustainable_yield:.1f} AF/month")
print(f"Capital monthly:   ${env.capital_monthly:,.0f}/month")
print()

# Test four fixed action fractions across full episode
for frac in [0.0, 0.25, 0.5, 1.0]:
    env.reset(seed=0)
    rewards  = []
    costs    = []
    risks    = []
    deficits = []
    desals   = []

    for t in range(env.H):
        obs, reward, term, trunc, info = env.step(np.array([frac], dtype=np.float32))
        rewards.append(reward)
        costs.append(info['monthly_cost'])
        risks.append(info['months_supply'])
        deficits.append(info['deficit'])
        desals.append(info['desal_release'])
        if term or trunc:
            break

    print(f"Action frac={frac:.2f}:")
    print(f"  Total reward:              {sum(rewards):.2f}")
    print(f"  Mean reward/step:          {np.mean(rewards):.4f}")
    print(f"  Mean cost/month:           ${np.mean(costs):,.0f}")
    print(f"  Mean desal release:        {np.mean(desals):.1f} AF/month")
    print(f"  Mean risk:                 {np.mean(risks):.1f} months")
    print(f"  Min risk:                  {np.min(risks):.1f} months")
    print(f"  Total deficit:             {sum(deficits):.1f} AF")
    print(f"  Steps below safety (3mo):  {sum(1 for r in risks if r < 3.0)}")
    print(f"  Steps with deficit:        {sum(1 for d in deficits if d > 0)}")
    print()

print("First 12 months detail at frac=0.5:")
env.reset(seed=0)
print(f"{'t':>3} {'desal':>8} {'sc':>8} {'sgi':>6} {'sswp':>6} {'risk':>6} {'cost':>10} {'reward':>8}")
for t in range(12):
    obs, reward, term, trunc, info = env.step(np.array([0.5], dtype=np.float32))
    print(f"{t:>3} {info['desal_release']:>8.1f} {info['sc']:>8.1f} "
          f"{info['sgi']:>6.1f} {info['sswp']:>6.1f} "
          f"{info['months_supply']:>6.1f} {info['monthly_cost']:>10,.0f} {reward:>8.4f}")
    if term or trunc:
        break