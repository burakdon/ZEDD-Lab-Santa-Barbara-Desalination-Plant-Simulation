#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_pareto_rl.py — Overlay NSGA-II Pareto front against RL agents.

Shows where the RL agents land relative to the full Pareto front
from the traditional optimization approach.

Usage (from project root):
    python3 src/compare_pareto_rl.py

Output:
    result/rl/figures/fig6_pareto_vs_rl.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'result', 'rl', 'trajectories')
FIG_DIR  = os.path.join(PROJECT_ROOT, 'result', 'rl', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f8f8',
    'axes.grid': True, 'grid.alpha': 0.4, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
})

# ── Load Pareto front ─────────────────────────────────────────────────────────
print("Loading Pareto front...")
pareto_path = os.path.join(
    PROJECT_ROOT, 'result', 'data', 'pareto',
    'pareto_pers87_sev0.83n_4_case_flexible_4mpd_36vessels.csv'
)
pareto = pd.read_csv(pareto_path)
pareto['cost_B'] = pareto['cost'] / 1e9

# Sort by cost for clean line
pareto = pareto.sort_values('cost_B')

print(f"  Pareto points: {len(pareto)}")
print(f"  Cost range:    ${pareto['cost'].min()/1e9:.2f}B — ${pareto['cost'].max()/1e9:.2f}B")
print(f"  Risk range:    {pareto['risk_months_supply'].min():.1f} — {pareto['risk_months_supply'].max():.1f} months")

# ── Load RL agent results ─────────────────────────────────────────────────────
print("Loading RL trajectories...")
cost_data = np.load(os.path.join(TRAJ_DIR, 'cost_only_drought.npz'))
safe_data = np.load(os.path.join(TRAJ_DIR, 'safe_drought.npz'))

rl_cost_only = {
    'cost_B':       float(cost_data['episode_total_cost']) / 1e9,
    'risk_months':  float(cost_data['episode_mean_risk']),
    'min_risk':     float(cost_data['episode_min_risk']),
}
rl_safe = {
    'cost_B':       float(safe_data['episode_total_cost']) / 1e9,
    'risk_months':  float(safe_data['episode_mean_risk']),
    'min_risk':     float(safe_data['episode_min_risk']),
}

print(f"  Cost-only RL: ${rl_cost_only['cost_B']:.2f}B | mean risk {rl_cost_only['risk_months']:.1f} mo")
print(f"  Safe RL:      ${rl_safe['cost_B']:.2f}B | mean risk {rl_safe['risk_months']:.1f} mo")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    'Fig 6 — NSGA-II Pareto Front vs RL Agents\n'
    'Traditional Optimization vs Reinforcement Learning  |  '
    'Flexible Tariff, 4MPD/36 Vessels, Severe Drought',
    fontweight='bold', fontsize=13
)

# ── Panel A: Mean risk comparison ─────────────────────────────────────────────
ax = axes[0]

ax.scatter(pareto['cost_B'], pareto['risk_months_supply'],
           color='#9E9E9E', alpha=0.5, s=40, zorder=3,
           label='NSGA-II Pareto front')

# Draw Pareto frontier line
ax.plot(pareto['cost_B'], pareto['risk_months_supply'],
        color='#9E9E9E', alpha=0.3, linewidth=1, zorder=2)

# RL agents
ax.scatter(rl_cost_only['cost_B'], rl_cost_only['risk_months'],
           color='#2196F3', s=250, zorder=5, marker='*',
           label=f"Cost-only RL  (${rl_cost_only['cost_B']:.2f}B, {rl_cost_only['risk_months']:.1f} mo)")
ax.scatter(rl_safe['cost_B'], rl_safe['risk_months'],
           color='#4CAF50', s=250, zorder=5, marker='*',
           label=f"Safe RL  (${rl_safe['cost_B']:.2f}B, {rl_safe['risk_months']:.1f} mo)")

# Annotations
ax.annotate('Cost-only RL',
            xy=(rl_cost_only['cost_B'], rl_cost_only['risk_months']),
            xytext=(rl_cost_only['cost_B'] - 0.08, rl_cost_only['risk_months'] - 3),
            fontsize=9, color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3'))
ax.annotate('Safe RL',
            xy=(rl_safe['cost_B'], rl_safe['risk_months']),
            xytext=(rl_safe['cost_B'] + 0.02, rl_safe['risk_months'] - 3),
            fontsize=9, color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50'))

ax.set_xlabel('Total Cost ($B)')
ax.set_ylabel('Mean Risk (months of supply)')
ax.set_title('Mean Risk vs Cost\n(higher risk = safer)')
ax.legend(fontsize=9)

# ── Panel B: Min risk comparison ──────────────────────────────────────────────
ax = axes[1]

# For Pareto we only have mean risk, not min risk
# So we show cost distribution of Pareto vs RL cost points
# alongside min risk for RL agents as horizontal reference lines

ax.scatter(pareto['cost_B'], pareto['risk_months_supply'],
           color='#9E9E9E', alpha=0.5, s=40, zorder=3,
           label='NSGA-II Pareto front (mean risk)')

ax.scatter(rl_cost_only['cost_B'], rl_cost_only['risk_months'],
           color='#2196F3', s=250, zorder=5, marker='*',
           label=f"Cost-only RL mean risk ({rl_cost_only['risk_months']:.1f} mo)")
ax.scatter(rl_safe['cost_B'], rl_safe['risk_months'],
           color='#4CAF50', s=250, zorder=5, marker='*',
           label=f"Safe RL mean risk ({rl_safe['risk_months']:.1f} mo)")

# Min risk as diamonds
ax.scatter(rl_cost_only['cost_B'], rl_cost_only['min_risk'],
           color='#2196F3', s=200, zorder=5, marker='D', alpha=0.7,
           label=f"Cost-only RL min risk ({rl_cost_only['min_risk']:.1f} mo)")
ax.scatter(rl_safe['cost_B'], rl_safe['min_risk'],
           color='#4CAF50', s=200, zorder=5, marker='D', alpha=0.7,
           label=f"Safe RL min risk ({rl_safe['min_risk']:.1f} mo)")

# Lines connecting mean to min for each agent
ax.plot([rl_cost_only['cost_B'], rl_cost_only['cost_B']],
        [rl_cost_only['min_risk'], rl_cost_only['risk_months']],
        color='#2196F3', linewidth=1.5, linestyle=':', alpha=0.7)
ax.plot([rl_safe['cost_B'], rl_safe['cost_B']],
        [rl_safe['min_risk'], rl_safe['risk_months']],
        color='#4CAF50', linewidth=1.5, linestyle=':', alpha=0.7)

ax.set_xlabel('Total Cost ($B)')
ax.set_ylabel('Risk (months of supply)')
ax.set_title('Mean vs Min Risk\n(★ = mean,  ◆ = worst-case minimum)')
ax.legend(fontsize=8)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, 'fig6_pareto_vs_rl.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved → {out_path}")