#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_rl.py — Generate all figures and animation for the RL challenge.

Produces:
    result/rl/figures/fig1_agent_comparison.png   — side by side metrics
    result/rl/figures/fig2_reservoir_storage.png  — storage trajectories
    result/rl/figures/fig3_desal_decisions.png    — desal operating decisions
    result/rl/figures/fig4_risk_profile.png       — risk over time
    result/rl/figures/fig5_alignment.png          — safety/cost tradeoff
    result/rl/animation/drought_animation.gif     — animated dashboard

Usage (from project root):
    python3 src/visualize_rl.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, PROJECT_ROOT)

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR  = os.path.join(PROJECT_ROOT, 'result', 'rl', 'figures')
ANIM_DIR = os.path.join(PROJECT_ROOT, 'result', 'rl', 'animation')
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'result', 'rl', 'trajectories')

for d in [FIG_DIR, ANIM_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'grid.linestyle':   '--',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.titleweight': 'bold',
})

COST_COLOR = '#2196F3'   # blue  — cost-only agent
SAFE_COLOR = '#4CAF50'   # green — safe agent
DANGER_COLOR = '#F44336' # red   — danger zone

# ── Load trajectories ─────────────────────────────────────────────────────────
print("Loading trajectories...")
cost_data = np.load(os.path.join(TRAJ_DIR, 'cost_only_drought.npz'))
safe_data = np.load(os.path.join(TRAJ_DIR, 'safe_drought.npz'))

# Unpack
c_sc   = cost_data['sc'];   s_sc   = safe_data['sc']
c_sgi  = cost_data['sgi'];  s_sgi  = safe_data['sgi']
c_sswp = cost_data['sswp']; s_sswp = safe_data['sswp']
c_desal = cost_data['desal']; s_desal = safe_data['desal']
c_cost  = cost_data['cost'];  s_cost  = safe_data['cost']
c_risk  = cost_data['risk'];  s_risk  = safe_data['risk']
c_deficit = cost_data['deficit']; s_deficit = safe_data['deficit']
desal_cap = float(cost_data['desal_capacity'])
demand    = cost_data['demand']

H      = len(c_sc)
time_y = np.arange(H) / 12.0   # convert months to years
SAFETY_THRESHOLD = 3.0

print(f"  Episode length: {H} months ({H//12} years)")
print(f"  Desal capacity: {desal_cap:.1f} AF/month")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Agent comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 1: Agent comparison...")

metrics = {
    'Total Cost\n($B)':        [float(cost_data['episode_total_cost']) / 1e9,
                                 float(safe_data['episode_total_cost']) / 1e9],
    'Mean Risk\n(months)':     [float(cost_data['episode_mean_risk']),
                                 float(safe_data['episode_mean_risk'])],
    'Min Risk\n(months)':      [float(cost_data['episode_min_risk']),
                                 float(safe_data['episode_min_risk'])],
    'Total Deficit\n(AF)':     [float(cost_data['episode_total_deficit']),
                                 float(safe_data['episode_total_deficit'])],
}

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Fig 1 — RL Agent Comparison: Cost-Only vs Safety-Constrained\n'
             'Santa Barbara Water Supply  |  Severe Drought Scenario',
             fontweight='bold', fontsize=13)

for ax, (metric, vals) in zip(axes, metrics.items()):
    bars = ax.bar(['Cost-only', 'Safe'], vals,
                  color=[COST_COLOR, SAFE_COLOR], alpha=0.85, width=0.5)
    ax.set_title(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{val:.2f}' if val < 10 else f'{val:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

cost_patch = mpatches.Patch(color=COST_COLOR, label='Cost-only agent')
safe_patch = mpatches.Patch(color=SAFE_COLOR, label='Safe agent')
fig.legend(handles=[cost_patch, safe_patch],
           loc='lower center', ncol=2, fontsize=11, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(os.path.join(FIG_DIR, 'fig1_agent_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_agent_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Reservoir storage over time
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 2: Reservoir storage...")

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.suptitle('Fig 2 — Reservoir Storage Over Time: Cost-Only vs Safe Agent\n'
             'Severe Drought Scenario  |  Cachuma + Gibraltar + SWP',
             fontweight='bold', fontsize=13)

for ax, sc, sgi, sswp, label, color in [
    (axes[0], c_sc, c_sgi, c_sswp, 'Cost-only agent', COST_COLOR),
    (axes[1], s_sc, s_sgi, s_sswp, 'Safe agent',      SAFE_COLOR),
]:
    total = sc + sgi + sswp
    ax.stackplot(time_y, sc, sgi, sswp,
                 labels=['Cachuma', 'Gibraltar', 'SWP'],
                 colors=['#08519c', '#3182bd', '#9ecae1'], alpha=0.8)
    ax.plot(time_y, total, color='black', linewidth=1.5, label='Total storage')
    ax.set_title(label)
    ax.set_ylabel('Storage (AF)')
    ax.legend(loc='upper right', fontsize=9, ncol=4)

axes[1].set_xlabel('Year')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_reservoir_storage.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_reservoir_storage.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Desalination decisions
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 3: Desalination decisions...")

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Fig 3 — Desalination Operating Decisions\n'
             'Cost-only agent minimises desal use  |  Safe agent runs desal more aggressively',
             fontweight='bold', fontsize=13)

for ax, desal, color, label in [
    (axes[0], c_desal, COST_COLOR, 'Cost-only agent'),
    (axes[1], s_desal, SAFE_COLOR, 'Safe agent'),
]:
    ax.fill_between(time_y, desal, alpha=0.6, color=color, label='Desal production')
    ax.axhline(desal_cap, color='black', linestyle='--', linewidth=1.5,
               label=f'Max capacity ({desal_cap:.0f} AF/mo)')
    ax.set_title(label)
    ax.set_ylabel('AF/month')
    ax.set_ylim(0, desal_cap * 1.15)
    ax.legend(loc='upper right', fontsize=9)

axes[1].set_xlabel('Year')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_desal_decisions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_desal_decisions.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Risk profile over time
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 4: Risk profile...")

fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle('Fig 4 — Supply Risk Over Time: Months of Water Remaining\n'
             'Safety threshold = 3 months  |  Below threshold = danger zone',
             fontweight='bold', fontsize=13)

ax.plot(time_y, c_risk, color=COST_COLOR, linewidth=2,
        label='Cost-only agent', alpha=0.9)
ax.plot(time_y, s_risk, color=SAFE_COLOR, linewidth=2,
        label='Safe agent', alpha=0.9)
ax.axhline(SAFETY_THRESHOLD, color=DANGER_COLOR, linestyle='--', linewidth=2,
           label=f'Safety threshold ({SAFETY_THRESHOLD} months)')
ax.fill_between(time_y, 0, SAFETY_THRESHOLD,
                color=DANGER_COLOR, alpha=0.08, label='Danger zone')

# Annotate minimum risk points
c_min_idx = np.argmin(c_risk)
s_min_idx = np.argmin(s_risk)
ax.annotate(f'Min: {c_risk[c_min_idx]:.1f} mo',
            xy=(time_y[c_min_idx], c_risk[c_min_idx]),
            xytext=(time_y[c_min_idx] + 2, c_risk[c_min_idx] + 3),
            fontsize=9, color=COST_COLOR,
            arrowprops=dict(arrowstyle='->', color=COST_COLOR))
ax.annotate(f'Min: {s_risk[s_min_idx]:.1f} mo',
            xy=(time_y[s_min_idx], s_risk[s_min_idx]),
            xytext=(time_y[s_min_idx] + 2, s_risk[s_min_idx] + 3),
            fontsize=9, color=SAFE_COLOR,
            arrowprops=dict(arrowstyle='->', color=SAFE_COLOR))

ax.set_xlabel('Year')
ax.set_ylabel('Months of supply remaining')
ax.set_ylim(0, max(c_risk.max(), s_risk.max()) * 1.15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_risk_profile.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_risk_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Alignment: cost vs safety tradeoff
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Fig 5: Alignment tradeoff...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fig 5 — The Alignment Problem: Cost Optimisation vs Safety\n'
             'An agent optimising cost alone accepts dangerous levels of supply risk',
             fontweight='bold', fontsize=13)

# Left: cumulative cost
ax = axes[0]
ax.plot(time_y, np.cumsum(c_cost) / 1e6, color=COST_COLOR, linewidth=2,
        label=f'Cost-only  (total: ${float(cost_data["episode_total_cost"])/1e9:.2f}B)')
ax.plot(time_y, np.cumsum(s_cost) / 1e6, color=SAFE_COLOR, linewidth=2,
        label=f'Safe  (total: ${float(safe_data["episode_total_cost"])/1e9:.2f}B)')
diff = (float(safe_data['episode_total_cost']) - float(cost_data['episode_total_cost'])) / 1e6
ax.set_title(f'Cumulative Cost\n(Safe agent pays ${diff:,.0f}M more)')
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative cost ($M)')
ax.legend(fontsize=9)

# Right: risk distribution histogram
ax = axes[1]
ax.hist(c_risk, bins=40, alpha=0.65, color=COST_COLOR,
        label=f'Cost-only  (min={c_risk.min():.1f} mo)')
ax.hist(s_risk, bins=40, alpha=0.65, color=SAFE_COLOR,
        label=f'Safe  (min={s_risk.min():.1f} mo)')
ax.axvline(SAFETY_THRESHOLD, color=DANGER_COLOR, linestyle='--', linewidth=2,
           label='Safety threshold')
ax.set_title('Distribution of Monthly Risk\n(months of supply remaining)')
ax.set_xlabel('Months of supply')
ax.set_ylabel('Count')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_alignment.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_alignment.png")


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATION — Drought dashboard
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating animation (this may take a minute)...")

# Subsample for animation — show every 6th month (every 0.5 years)
STEP = 6
frames = list(range(0, H, STEP))
N_FRAMES = len(frames)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_storage = fig.add_subplot(gs[0, :])   # full width — storage
ax_desal   = fig.add_subplot(gs[1, 0])   # desal decisions
ax_risk    = fig.add_subplot(gs[1, 1])   # risk meter
ax_cost    = fig.add_subplot(gs[2, 0])   # cumulative cost
ax_info    = fig.add_subplot(gs[2, 1])   # info panel

DARK_BG    = '#1a1a2e'
PANEL_BG   = '#16213e'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#2a2a4a'

for ax in [ax_storage, ax_desal, ax_risk, ax_cost, ax_info]:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.5)

fig.suptitle('Santa Barbara Water Supply — RL Agent Comparison\nSevere Drought Scenario',
             color=TEXT_COLOR, fontsize=14, fontweight='bold')


def animate(frame_idx):
    t = frames[frame_idx]
    yr = t / 12.0

    for ax in [ax_storage, ax_desal, ax_risk, ax_cost, ax_info]:
        ax.cla()
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, alpha=0.5)

    t1 = t + 1

    # ── Storage panel ─────────────────────────────────────────────────────────
    ax_storage.stackplot(
        time_y[:t1],
        c_sc[:t1], c_sgi[:t1], c_sswp[:t1],
        colors=['#1565C0', '#1976D2', '#42A5F5'],
        alpha=0.7, labels=['Cachuma (cost)', 'Gibraltar (cost)', 'SWP (cost)']
    )
    ax_storage.stackplot(
        time_y[:t1],
        s_sc[:t1], s_sgi[:t1], s_sswp[:t1],
        colors=['#2E7D32', '#388E3C', '#81C784'],
        alpha=0.4, labels=['Cachuma (safe)', 'Gibraltar (safe)', 'SWP (safe)']
    )
    ax_storage.set_title(f'Total Reservoir Storage  |  Year {yr:.1f}', color=TEXT_COLOR)
    ax_storage.set_ylabel('AF', color=TEXT_COLOR)
    ax_storage.set_xlim(0, H / 12)
    ax_storage.axvline(yr, color='white', linewidth=1.5, alpha=0.7)

    # ── Desal decisions ───────────────────────────────────────────────────────
    ax_desal.fill_between(time_y[:t1], c_desal[:t1],
                           color=COST_COLOR, alpha=0.7, label='Cost-only')
    ax_desal.fill_between(time_y[:t1], s_desal[:t1],
                           color=SAFE_COLOR, alpha=0.5, label='Safe')
    ax_desal.axhline(desal_cap, color='white', linestyle='--', linewidth=1,
                     label='Max capacity')
    ax_desal.set_title('Desalination Production', color=TEXT_COLOR)
    ax_desal.set_ylabel('AF/month', color=TEXT_COLOR)
    ax_desal.set_xlim(0, H / 12)
    ax_desal.set_ylim(0, desal_cap * 1.2)
    ax_desal.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
    ax_desal.axvline(yr, color='white', linewidth=1.5, alpha=0.7)

    # ── Risk meter ────────────────────────────────────────────────────────────
    ax_risk.plot(time_y[:t1], c_risk[:t1], color=COST_COLOR,
                 linewidth=2, label=f'Cost-only  (now: {c_risk[t]:.1f} mo)')
    ax_risk.plot(time_y[:t1], s_risk[:t1], color=SAFE_COLOR,
                 linewidth=2, label=f'Safe  (now: {s_risk[t]:.1f} mo)')
    ax_risk.axhline(SAFETY_THRESHOLD, color=DANGER_COLOR,
                    linestyle='--', linewidth=1.5)
    ax_risk.fill_between(time_y[:t1], 0, SAFETY_THRESHOLD,
                          color=DANGER_COLOR, alpha=0.15)
    ax_risk.set_title('Supply Risk (months remaining)', color=TEXT_COLOR)
    ax_risk.set_ylabel('Months', color=TEXT_COLOR)
    ax_risk.set_xlim(0, H / 12)
    ax_risk.set_ylim(0, max(c_risk.max(), s_risk.max()) * 1.15)
    ax_risk.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
    ax_risk.axvline(yr, color='white', linewidth=1.5, alpha=0.7)

    # ── Cumulative cost ───────────────────────────────────────────────────────
    ax_cost.plot(time_y[:t1], np.cumsum(c_cost[:t1]) / 1e6,
                 color=COST_COLOR, linewidth=2,
                 label=f'Cost-only  (${np.sum(c_cost[:t1])/1e6:,.0f}M)')
    ax_cost.plot(time_y[:t1], np.cumsum(s_cost[:t1]) / 1e6,
                 color=SAFE_COLOR, linewidth=2,
                 label=f'Safe  (${np.sum(s_cost[:t1])/1e6:,.0f}M)')
    ax_cost.set_title('Cumulative Operating Cost', color=TEXT_COLOR)
    ax_cost.set_ylabel('$M', color=TEXT_COLOR)
    ax_cost.set_xlim(0, H / 12)
    ax_cost.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
    ax_cost.axvline(yr, color='white', linewidth=1.5, alpha=0.7)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.axis('off')
    month_name = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec'][t % 12]

    c_risk_now = c_risk[t]
    s_risk_now = s_risk[t]
    c_risk_color = '#F44336' if c_risk_now < SAFETY_THRESHOLD else '#4CAF50'
    s_risk_color = '#F44336' if s_risk_now < SAFETY_THRESHOLD else '#4CAF50'

    info_text = (
        f"Month {t+1} of {H}  |  {month_name} Year {int(yr)+1}\n\n"
        f"COST-ONLY AGENT\n"
        f"  Desal:  {c_desal[t]:6.1f} AF/mo\n"
        f"  Risk:   {c_risk_now:6.1f} months\n"
        f"  Cost:   ${c_cost[t]/1e3:6.0f}K/mo\n\n"
        f"SAFE AGENT\n"
        f"  Desal:  {s_desal[t]:6.1f} AF/mo\n"
        f"  Risk:   {s_risk_now:6.1f} months\n"
        f"  Cost:   ${s_cost[t]/1e3:6.0f}K/mo\n\n"
        f"Safety threshold: {SAFETY_THRESHOLD} months"
    )
    ax_info.text(0.05, 0.95, info_text,
                 transform=ax_info.transAxes,
                 fontsize=10, verticalalignment='top',
                 fontfamily='monospace',
                 color=TEXT_COLOR,
                 bbox=dict(boxstyle='round', facecolor=PANEL_BG,
                           edgecolor=GRID_COLOR, alpha=0.8))

    return []


anim = FuncAnimation(
    fig, animate,
    frames     = N_FRAMES,
    interval   = 100,
    blit       = False,
    repeat     = False,
)

anim_path = os.path.join(ANIM_DIR, 'drought_animation.gif')
writer = PillowWriter(fps=10)
anim.save(anim_path, writer=writer)
plt.close()
print(f"  Saved drought_animation.gif  ({N_FRAMES} frames)")

print(f"\n{'='*60}")
print("ALL VISUALISATIONS COMPLETE")
print(f"{'='*60}")
print(f"Figures → result/rl/figures/")
print(f"  fig1_agent_comparison.png")
print(f"  fig2_reservoir_storage.png")
print(f"  fig3_desal_decisions.png")
print(f"  fig4_risk_profile.png")
print(f"  fig5_alignment.png")
print(f"Animation → result/rl/animation/")
print(f"  drought_animation.gif")