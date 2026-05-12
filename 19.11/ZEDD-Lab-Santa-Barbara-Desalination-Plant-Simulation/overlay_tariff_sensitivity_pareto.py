#!/usr/bin/env python3
"""
Overlay Pareto fronts for all ``supply_curve_tariff_sensitivity`` tariff cases.

Discovers CSVs saved as:
  result/data/pareto/pareto_<drought>_case_supply_curve_tariff_sensitivity_hour*_day*_year*_baseline|flexible_3mpd_30vessels.csv

Two panels: baseline vs flexible. Each (hour, day, year) combination gets one color
from a shared colormap; curves are sorted by cost and drawn as translucent lines.

Usage (from project root):
  python3 overlay_tariff_sensitivity_pareto.py --drought pers87_sev0.83n_4
  python3 overlay_tariff_sensitivity_pareto.py --drought pers87_sev0.83n_4 \\
      --out result/plots/pareto/tariff_overlay.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

CASE_RE = re.compile(
    r"^supply_curve_tariff_sensitivity_hour(\d+)_day(\d+)_year(\d+)_(baseline|flexible)_3mpd_30vessels$"
)


def discover_csvs(drought: str, pareto_dir: str) -> List[str]:
    pat = os.path.join(
        pareto_dir,
        f"pareto_{drought}_case_supply_curve_tariff_sensitivity_*.csv",
    )
    paths = sorted(glob.glob(pat))
    return [p for p in paths if CASE_RE.search(os.path.basename(p).split("case_", 1)[-1][:-4])]


def parse_case_filename(case_fn: str) -> Optional[Tuple[int, int, int, str]]:
    m = CASE_RE.match(case_fn)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def build_color_lookup(keys: List[Tuple[int, int, int]], cmap_name: str):
    """Map each unique (hour, day, year) to a color using ``tab20`` / qualitative or ``viridis``."""
    uniq = sorted(set(keys))
    n = len(uniq)
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max(n - 1, 1))
    lut = {k: cmap(norm(i)) for i, k in enumerate(uniq)}
    return lut, uniq


def plot_overlay(
    drought: str,
    paths: List[str],
    out: str,
    cmap_name: str,
    title: str | None,
):
    by_panel: Dict[str, List[Tuple[str, Tuple[int, int, int, str]]]] = {
        "baseline": [],
        "flexible": [],
    }
    all_keys: List[Tuple[int, int, int]] = []

    for path in paths:
        base = os.path.basename(path)
        # pareto_<drought>_case_<casefn>.csv
        if f"pareto_{drought}_case_" not in base or not base.endswith(".csv"):
            continue
        case_fn = base[len(f"pareto_{drought}_case_") : -4]
        parsed = parse_case_filename(case_fn)
        if parsed is None:
            print(f"WARN: skip (unparsed name): {base}")
            continue
        h, d, y, tariff = parsed
        if tariff not in by_panel:
            continue
        by_panel[tariff].append((path, (h, d, y, tariff)))
        all_keys.append((h, d, y))

    color_lut, uniq_keys = build_color_lookup(all_keys, cmap_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, panel in zip(axes, ("baseline", "flexible")):
        items = by_panel[panel]
        if not items:
            ax.text(
                0.5,
                0.5,
                "No Pareto CSVs for this panel.\nRun the tariff batch for this drought.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="#555",
            )
            ax.set_title(f"{panel.capitalize()} (0 fronts)")
        else:
            for path, (h, d, y, _) in sorted(
                items, key=lambda x: (x[1][0], x[1][1], x[1][2])
            ):
                df = pd.read_csv(path)
                if df.empty:
                    continue
                if "cost" not in df.columns or "risk_months_supply" not in df.columns:
                    print(f"WARN: missing columns in {path}")
                    continue
                df = df.sort_values("cost")
                col = color_lut[(h, d, y)]
                ax.plot(
                    df["cost"].values,
                    df["risk_months_supply"].values,
                    color=col,
                    alpha=0.45,
                    linewidth=1.4,
                    solid_capstyle="round",
                )
            ax.set_title(f"{panel.capitalize()} ({len(items)} fronts)")
        ax.set_xlabel("Mean cost (objective J1)")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Months of supply in storage (risk proxy, higher is safer)")

    vmax = max(len(uniq_keys) - 1, 1)
    sm = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_name),
        norm=mcolors.Normalize(vmin=0, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), fraction=0.035, pad=0.04)
    n_ticks = min(9, len(uniq_keys))
    tick_idx = (
        np.linspace(0, vmax, n_ticks).astype(int)
        if len(uniq_keys) > 1
        else np.array([0], dtype=int)
    )
    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels(
        [f"h{uniq_keys[i][0]} d{uniq_keys[i][1]} y{uniq_keys[i][2]}" for i in tick_idx]
    )
    cbar.set_label("(hour, day, year) — sorted scenario index → color")

    ttl = title or (
        f"Tariff sensitivity — Pareto overlays ({drought})\n"
        f"Color = (hour, day, year); {len(by_panel['baseline']) + len(by_panel['flexible'])} fronts"
    )
    fig.suptitle(ttl, fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.12, wspace=0.12)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved: {out} (panels baseline={len(by_panel['baseline'])}, flexible={len(by_panel['flexible'])})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drought", default="pers87_sev0.83n_4")
    ap.add_argument("--pareto-dir", default="result/data/pareto")
    ap.add_argument(
        "--out",
        default="result/plots/pareto/tariff_sensitivity_overlay_baseline_flexible.png",
    )
    ap.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name for (hour, day, year) index",
    )
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    paths = discover_csvs(args.drought, args.pareto_dir)
    if not paths:
        print(
            f"No Pareto CSVs found under {args.pareto_dir!r} for drought {args.drought!r}. "
            "Run: python3 run_tariff_sensitivity_batch.py --drought ... --cases all"
        )
        return 1

    plot_overlay(args.drought, paths, args.out, args.cmap, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
