#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
water_env.py — Gymnasium environment wrapping the Santa Barbara water simulation.

The RL agent directly outputs a desalination release fraction each month.
The RBF policy machinery from sim_individual.py is bypassed entirely.

MDP definition:
  State  : [norm_storage, month_of_year, norm_sc, norm_sgi, norm_sswp]
  Action : desal_release_fraction in [0, 1]  (scaled by desal_capacity internally)
  Reward : survival bonus - cost_penalty - deficit_penalty - safety_penalty
  Episode: one full hydrological scenario (H months, typically 1200 months)

Usage:
    from water_env import WaterEnv
    env = WaterEnv(case_number='basetariff_flexible/4mpd_36vessels',
                   drought_type='pers87_sev0.83n_4')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from cachuma_lake import Cachuma
from gibraltar_lake import Gibraltar
from swp_lake import SWP
from cost_curve_loader import CostCurveLoader

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def decorator(func):
            return func
        return decorator


@njit
def _mass_balance(s, u, n, s_max):
    r  = max(0.0, min(u, s))
    s_ = s + n - r
    s_ = max(0.0, min(s_, s_max))
    return s_, r


class WaterEnv(gym.Env):
    """
    Santa Barbara water supply environment for RL.

    Reward structure (per timestep):
      +1.0                         survival bonus (always)
      - cost / cost_scale          operational cost penalty (small)
      - 50.0                       if deficit > 0 (hard supply failure)
      - safety_penalty             if months_supply < safety_threshold_months
                                   scaled by how far below threshold

    This ensures:
      - Zero desal (frac=0) is heavily punished when it causes deficits
      - The safe agent is additionally punished for low storage risk
      - Cost minimisation is a secondary objective, not the primary one

    Parameters
    ----------
    case_number : str
        Cost curve case id, e.g. 'basetariff_flexible/4mpd_36vessels'.
    drought_type : str
        Hydrological scenario identifier.
    scenario_idx : int or None
        Fixed scenario index. None = random each episode.
    safety_threshold_months : float
        Months-of-supply below which safety penalty applies.
    safety_penalty : float
        Penalty per timestep below safety threshold (for safe agent).
        Set to 0.0 for cost-only agent.
    cost_scale : float
        Divisor to normalise monthly cost into reward range.
    data_dir : str
        Path to data directory.
    cost_curves_dir : str
        Path to cost_curves directory.
    randomise_init_storage : bool
        Randomise initial reservoir levels during training.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        case_number: str = "basetariff_flexible/4mpd_36vessels",
        drought_type: str = "pers87_sev0.83n_4",
        scenario_idx: int = None,
        safety_threshold_months: float = 3.0,
        safety_penalty: float = 0.0,
        cost_scale: float = 1e6,
        data_dir: str = "data",
        cost_curves_dir: str = "cost_curves",
        randomise_init_storage: bool = True,
    ):
        super().__init__()

        self.case_number             = case_number
        self.drought_type            = drought_type
        self.scenario_idx            = scenario_idx
        self.safety_threshold_months = safety_threshold_months
        self.safety_penalty          = safety_penalty
        self.cost_scale              = cost_scale
        self.data_dir                = data_dir
        self.randomise_init_storage  = randomise_init_storage

        # ── Load hydrology & demand ────────────────────────────────────────────
        self.cachuma   = Cachuma(drought_type)
        self.gibraltar = Gibraltar(drought_type)
        self.swp       = SWP(drought_type)
        self.H         = self.gibraltar.H

        demand_path  = os.path.join(data_dir, "d12_predrought.txt")
        self.demand  = np.loadtxt(demand_path)
        mission_path = os.path.join(data_dir, f"mission_{drought_type}.txt")
        self.mds     = np.loadtxt(mission_path)

        self.sustainable_yield     = 1250.0 / 12.0
        self.storage_normalization = 35000.0

        # ── Cost curves ───────────────────────────────────────────────────────
        self.cost_loader     = CostCurveLoader(cost_curves_dir)
        self.cost_curve_data = self.cost_loader.load_cost_curve(case_number)
        self.desal_capacity  = self.cost_loader.get_max_production(case_number)
        self.capital_monthly = self.cost_loader.get_capital_cost_amortized(
            case_number, amortization_years=30, period="monthly"
        )

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low   = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high  = np.array([2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype = np.float32,
        )

        self.action_space = spaces.Box(
            low   = np.array([0.0], dtype=np.float32),
            high  = np.array([1.0], dtype=np.float32),
            dtype = np.float32,
        )

        self.t     = 0
        self.sc    = 0.0
        self.sgi   = 0.0
        self.sswp  = 0.0
        self._scenario = 0

        self.ncs   = self.cachuma.inflow
        self.ngis  = self.gibraltar.inflow
        self.nswps = self.swp.inflow
        self.nsim  = self.ncs.shape[0]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_obs(self):
        norm_total = (self.sc + self.sgi + self.sswp) / self.storage_normalization
        moy        = (self.t % 12) / 12.0
        norm_sc    = self.sc   / self.cachuma.smax
        norm_sgi   = self.sgi  / self.gibraltar.smax
        norm_sswp  = self.sswp / self.swp.smax
        return np.array(
            [norm_total, moy, norm_sc, norm_sgi, norm_sswp],
            dtype=np.float32
        )

    def _months_of_supply(self, sc, sgi, sswp):
        net_demand = np.mean(self.demand) - self.desal_capacity - self.sustainable_yield
        if net_demand <= 0:
            return np.inf
        return (sc + sgi + sswp) / net_demand

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.scenario_idx is not None:
            self._scenario = self.scenario_idx
        else:
            self._scenario = self.np_random.integers(0, self.nsim)

        if self.randomise_init_storage:
            self.sc   = float(self.np_random.uniform(0.3, 1.0) * self.cachuma.smax)
            self.sgi  = float(self.np_random.uniform(0.3, 1.0) * self.gibraltar.smax)
            self.sswp = float(self.np_random.uniform(0.3, 1.0) * self.swp.smax)
        else:
            self.sc   = float(self.cachuma.s0)
            self.sgi  = float(self.gibraltar.s0)
            self.sswp = float(self.swp.s0)

        self.t = 0

        self._sc_hist      = [self.sc]
        self._sgi_hist     = [self.sgi]
        self._sswp_hist    = [self.sswp]
        self._desal_hist   = []
        self._cost_hist    = []
        self._risk_hist    = []
        self._deficit_hist = []
        self._reward_hist  = []

        return self._get_obs(), {}

    def step(self, action):
        t  = self.t
        s  = self._scenario
        H  = self.H

        nc   = self.ncs[s, :]
        ngi  = self.ngis[s, :]
        nswp = self.nswps[s, :]
        md   = self.mds[s, :]

        # Cachuma annual allocation in October
        if t % 12 == 9:
            self.sc = self.sc * self.cachuma.carryover + \
                      (1 - self.cachuma.carryover) * 0.3
            nc_ = float(nc[int((t - 9) / 12)])
        else:
            nc_ = 0.0

        # SWP annual allocation in May
        nswp_ = float(nswp[int((t - 4) / 12)]) if t % 12 == 4 else 0.0

        # ── Desalination action ───────────────────────────────────────────────
        frac          = float(np.clip(action[0], 0.0, 1.0))
        desal_release = frac * self.desal_capacity

        # ── Surface water demand ──────────────────────────────────────────────
        dem = float(self.demand[t % 12])
        d   = max(0.0, dem - desal_release - float(md[t]) - self.sustainable_yield)

        # Proportional allocation across reservoirs
        SS   = 1e-4 + self.sc + self.sgi + self.sswp
        uc   = self.sc   / SS
        ugi  = self.sgi  / SS
        uswp = self.sswp / SS

        while uswp * d > self.swp.max_release:
            uswp -= 0.05
            uc   += 0.04
            ugi  += 0.01

        # ── Mass balance ──────────────────────────────────────────────────────
        sc_new,   r_c   = _mass_balance(self.sc,   uc   * d, nc_,           self.cachuma.smax)
        sgi_new,  r_gi  = _mass_balance(self.sgi,  ugi  * d, float(ngi[t]), self.gibraltar.smax)
        sswp_new, r_swp = _mass_balance(self.sswp, uswp * d, nswp_,         self.swp.smax)

        self.sc   = sc_new
        self.sgi  = sgi_new
        self.sswp = sswp_new

        # ── Cost calculation ──────────────────────────────────────────────────
        month     = t % 12
        is_summer = 4 <= month <= 9

        if desal_release > 1e-6:
            elec_cost, fixed_cost = self.cost_loader.get_cost_for_production(
                self.case_number, desal_release, is_summer
            )
            labor_cost = self.cost_loader.get_labor_cost(self.case_number)
            base_cost  = fixed_cost + elec_cost + labor_cost
        else:
            season_key = "summer" if is_summer else "winter"
            fixed_cost = float(self.cost_curve_data[season_key]["fixed_cost"][0])
            base_cost  = fixed_cost

        monthly_cost = base_cost + self.capital_monthly

        # ── Deficit ──────────────────────────────────────────────────────────
        deficit = max(
            0.0,
            dem - r_swp - r_c - r_gi - float(md[t]) - desal_release - self.sustainable_yield
        )
        if deficit < 1e-4:
            deficit = 0.0

        # ── Risk ─────────────────────────────────────────────────────────────
        months_supply = self._months_of_supply(self.sc, self.sgi, self.sswp)

        # ── Reward ────────────────────────────────────────────────────────────
        # Survival bonus — agent gets this every timestep just for existing
        reward = 1.0

        # Cost penalty — secondary objective, scaled small
        reward -= monthly_cost / self.cost_scale

        # Deficit penalty — catastrophic, non-negotiable
        if deficit > 0:
            reward -= 50.0

        # Safety penalty — only applied for safe agent (safety_penalty > 0)
        # Scaled by how far below threshold, so deeper = worse
        if months_supply < self.safety_threshold_months and self.safety_penalty > 0:
            shortfall = self.safety_threshold_months - months_supply
            reward -= self.safety_penalty * shortfall

        # ── Logging ──────────────────────────────────────────────────────────
        self._sc_hist.append(self.sc)
        self._sgi_hist.append(self.sgi)
        self._sswp_hist.append(self.sswp)
        self._desal_hist.append(desal_release)
        self._cost_hist.append(monthly_cost)
        self._risk_hist.append(months_supply)
        self._deficit_hist.append(deficit)
        self._reward_hist.append(reward)

        # ── Termination ──────────────────────────────────────────────────────
        self.t    += 1
        terminated = self.t >= H
        truncated  = False

        info = {
            "t":             t,
            "desal_release": desal_release,
            "desal_frac":    frac,
            "monthly_cost":  monthly_cost,
            "deficit":       deficit,
            "months_supply": months_supply,
            "sc":            self.sc,
            "sgi":           self.sgi,
            "sswp":          self.sswp,
            "r_c":           r_c,
            "r_gi":          r_gi,
            "r_swp":         r_swp,
        }

        if terminated:
            info["episode_total_cost"]    = float(np.sum(self._cost_hist))
            info["episode_mean_risk"]     = float(np.mean(self._risk_hist))
            info["episode_min_risk"]      = float(np.min(self._risk_hist))
            info["episode_total_deficit"] = float(np.sum(self._deficit_hist))
            info["episode_total_reward"]  = float(np.sum(self._reward_hist))

        return self._get_obs(), float(reward), terminated, truncated, info

    def get_trajectory(self):
        return {
            "sc":             np.array(self._sc_hist[:-1]),
            "sgi":            np.array(self._sgi_hist[:-1]),
            "sswp":           np.array(self._sswp_hist[:-1]),
            "desal":          np.array(self._desal_hist),
            "cost":           np.array(self._cost_hist),
            "risk":           np.array(self._risk_hist),
            "deficit":        np.array(self._deficit_hist),
            "reward":         np.array(self._reward_hist),
            "desal_capacity": self.desal_capacity,
            "demand":         self.demand,
        }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running WaterEnv sanity check...")

    env = WaterEnv(
        case_number            = "basetariff_flexible/4mpd_36vessels",
        drought_type           = "pers87_sev0.83n_4",
        cost_curves_dir        = "cost_curves",
        data_dir               = "data",
        scenario_idx           = 0,
        randomise_init_storage = False,
    )

    obs, _ = env.reset(seed=42)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space:      {env.action_space}")
    print(f"  Episode length:    {env.H} months ({env.H // 12} years)")
    print(f"  Desal capacity:    {env.desal_capacity:.1f} AF/month")
    print(f"  Initial obs:       {obs}")

    total_reward = 0.0
    for step in range(env.H):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    print(f"\n  Random policy episode complete:")
    print(f"    Total reward:  {total_reward:.2f}")
    print(f"    Total cost:    ${info['episode_total_cost']:,.0f}")
    print(f"    Mean risk:     {info['episode_mean_risk']:.1f} months")
    print(f"    Min risk:      {info['episode_min_risk']:.1f} months")
    print(f"    Total deficit: {info['episode_total_deficit']:.1f} AF")
    print(f"\n✅ WaterEnv sanity check passed")