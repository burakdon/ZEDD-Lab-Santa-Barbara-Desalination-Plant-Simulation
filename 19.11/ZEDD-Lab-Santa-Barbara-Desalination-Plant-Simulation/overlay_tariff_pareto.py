#!/usr/bin/env python3
"""
Overlay Pareto fronts for all tariff-sensitivity cases (supply_curve_tariff_sensitivity).

Expects CSV files produced by run_tariff_sensitivity_batch.py / run_all_cases.py:
  result/data/pareto/pareto_<drought>_case_supply_curve_tariff_sensitivity_*.csv

Usage (from project root):
  python3 overlay_tariff_pareto.py --drought pers87_sev0.83n_4
  python3 overlay_tariff_pareto.py --drought pers87_sev0.83n_4 \\
      --out result/plots/pareto/overlay_tariff_all_pers87.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd


CASE_RE = re.compile(
    r"^supply_curve_tariff_sensitivity_hour(\d+)_day(\d+)_year(\d+)_(baseline|flexible)_(.+)$"
)


def discover_csvs(drought: str, pareto_dir: str) -> List[str]:
    pat = os.path.join(
        pareto_dir, f"pareto_{drought}_case_supply_curve_tariff_sensitivity*.csv"
    )
    paths = sorted(glob.glob(pat))
    return [p for p in paths if os.path.isfile(p)]


def parse_case_filename(case_fn: str) -> Optional[Tuple[int, int, int, str, str]]:
    """
    case_fn: filename stem after ``pareto_<drought>_case_`` and before ``.csv``
    Returns (hour, day, year, mode, stem) or None if not a tariff case.
    """
    m = CASE_RE.match(case_fn)
    if not m:
        return None
    h, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    mode, stem = m.group(4), m.group(5)
    return h, d, y, mode, stem


def scenario_key(h: int, d: int, y: int) -> Tuple[int, int, int]:
    return (h, d, y)


def plot_overlay(
    drought: str,
    paths: List[str],
    out: str,
    title: Optional[str],
    alpha: float,
    markersize: float,
) -> None:
    rows: List[Tuple[str, int, int, int, str, str, pd.DataFrame]] = []
    for p in paths:
        base = os.path.basename(p)
        prefix = f"pareto_{drought}_case_"
        if not base.startswith(prefix) or not base.endswith(".csv"):
            continue
        case_fn = base[len(prefix) : -len(".csv")]
        parsed = parse_case_filename(case_fn)
        if parsed is None:
            continue
        h, d, y, mode, stem = parsed
        df = pd.read_csv(p)
        if "cost" not in df.columns or "risk_months_supply" not in df.columns:
            print(f"WARN: skip (bad columns): {p}")
            continue
        rows.append((p, h, d, y, mode, stem, df))

    if not rows:
        raise SystemExit(
            f"No valid tariff Pareto CSVs found for drought={drought!r}. "
            f"Expected files like: result/data/pareto/pareto_{drought}_case_supply_curve_tariff_sensitivity_*.csv"
        )

    stems = sorted({r[5] for r in rows})
    if len(stems) > 1:
        print(f"NOTE: multiple curve stems in data: {stems} (colors keyed by hour/day/year only)")

    scenarios = sorted({scenario_key(r[1], r[2], r[3]) for r in rows})
    n_sc = len(scenarios)
    cmap = cm.get_cmap("viridis")
    scen_colors: Dict[Tuple[int, int, int], tuple] = {
        s: cmap(i / max(1, n_sc - 1)) for i, s in enumerate(scenarios)
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, mode in zip(axes, ("baseline", "flexible")):
        subset = [r for r in rows if r[4] == mode]
        for _p, h, d, y, _mmode, _stem, df in subset:
            c = scen_colors[(h, d, y)]
            lab = f"h{h} d{d} y{y}"
            ax.scatter(
                df["cost"],
                df["risk_months_supply"],
                s=markersize**2,
                c=[c],
                label=lab,
                alpha=alpha,
                edgecolors="none",
            )
        ax.set_title(mode.capitalize())
        ax.set_xlabel("cost (simulator units)")
        ax.grid(True, alpha=0.35)

    axes[0].set_ylabel("# demand months left in storage (risk)")
    fig.suptitle(
        title
        or f"Tariff sensitivity Pareto overlays — {drought} ({len(rows)} curves)",
        fontsize=13,
    )

    handles = []
    labels = []
    for s in scenarios:
        h, d, y = s
        color = scen_colors[s]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=6,
                linestyle="None",
            )
        )
        labels.append(f"h{h} d{d} y{y}")

    ncol = int(min(9, max(3, (n_sc + 2) // 6)))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=7,
        frameon=True,
        title="Tariff scenario (Hours / Days / Years)",
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}  (used {len(rows)} CSV files)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Overlay Pareto fronts for tariff sensitivity cases")
    ap.add_argument("--drought", required=True)
    ap.add_argument(
        "--pareto-dir",
        default="result/data/pareto",
        help="Directory containing pareto_*.csv files",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PNG (default: result/plots/pareto/overlay_tariff_sensitivity_<drought>.png)",
    )
    ap.add_argument("--title", default=None)
    ap.add_argument("--alpha", type=float, default=0.45, help="Point transparency (0-1)")
    ap.add_argument("--markersize", type=float, default=4.0)
    args = ap.parse_args()

    paths = discover_csvs(args.drought, args.pareto_dir)
    if not paths:
        print(
            f"No CSVs matched: {os.path.join(args.pareto_dir, f'pareto_{args.drought}_case_supply_curve_tariff_sensitivity*.csv')}",
            file=sys.stderr,
        )
        return 1

    out = args.out or os.path.join(
        "result", "plots", "pareto", f"overlay_tariff_sensitivity_{args.drought}.png"
    )
    plot_overlay(args.drought, paths, out, args.title, args.alpha, args.markersize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
