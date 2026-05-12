#!/usr/bin/env python3
"""
Overlay Pareto fronts for all ``supply_curve_tariff_sensitivity`` tariff cases.

Discovers CSVs saved as:
  result/data/pareto/pareto_<drought>_case_supply_curve_tariff_sensitivity_hour*_day*_year*_baseline|flexible_3mpd_30vessels.csv

Two panels: **baseline** (left) and **flexible** (right), matching the style of
``overlay_pareto_fixed_vs_flex.py``:

- **Color** encodes **peak hours** (2 / 5 / 8) using the first three Matplotlib tab10 colors
  (same family as other project overlays).
- **Linestyle** encodes **day** (peak/off-peak price ratio level: 1, 5, 10): solid, dashed, dotted.
- **Line width** encodes **year** (summer/winter charge ratio level: 1, 2, 5).
- **Flexible** panel uses the same mapping but each color is passed through ``mute_color()``
  (desaturated / lighter), analogous to flexible curves elsewhere in the repo.

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
import pandas as pd
from matplotlib.lines import Line2D

# First three tab10 colors (default Matplotlib cycle) — same spirit as overlay_pareto*.py
HOUR_COLORS = {
    2: "#1f77b4",
    5: "#ff7f0e",
    8: "#2ca02c",
}
DAY_LINESTYLE = {
    1: "-",
    5: "--",
    10: ":",
}
YEAR_LINEWIDTH = {
    1: 1.05,
    2: 1.45,
    5: 1.9,
}


def mute_color(
    color_hex: str, saturation_reduction: float = 0.5, brightness_increase: float = 0.1
) -> str:
    """Match ``overlay_pareto_fixed_vs_flex.mute_color`` (flexible / muted curves)."""
    rgb = mcolors.hex2color(color_hex)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] = max(0.0, hsv[1] - saturation_reduction)
    hsv[2] = min(1.0, hsv[2] + brightness_increase)
    rgb_muted = mcolors.hsv_to_rgb(hsv)
    return mcolors.rgb2hex(rgb_muted)


def style_for_scenario(h: int, d: int, y: int, panel: str) -> Tuple[str, str, float]:
    """Return (color_hex, linestyle, linewidth)."""
    col = HOUR_COLORS.get(h, "#7f7f7f")
    if panel == "flexible":
        col = mute_color(col)
    ls = DAY_LINESTYLE.get(d, "-")
    lw = YEAR_LINEWIDTH.get(y, 1.2)
    return col, ls, lw


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


def plot_overlay(
    drought: str,
    paths: List[str],
    out: str,
    title: str | None,
):
    by_panel: Dict[str, List[Tuple[str, Tuple[int, int, int, str]]]] = {
        "baseline": [],
        "flexible": [],
    }

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

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
                col, ls, lw = style_for_scenario(h, d, y, panel)
                ax.plot(
                    df["cost"].values,
                    df["risk_months_supply"].values,
                    color=col,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=0.62,
                    solid_capstyle="round",
                )
            ax.set_title(f"{panel.capitalize()} ({len(items)} fronts)")
        ax.set_xlabel("Mean cost (objective J1)")
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Months of supply in storage (risk proxy, higher is safer)")

    leg_hour = [
        Line2D(
            [0],
            [0],
            color=HOUR_COLORS[h],
            lw=2.4,
            linestyle="-",
            label=f"Peak hours = {h}",
        )
        for h in sorted(HOUR_COLORS)
    ]
    leg_day = [
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=2.0,
            linestyle=DAY_LINESTYLE[d],
            label=f"Day ratio level = {d}",
        )
        for d in sorted(DAY_LINESTYLE)
    ]
    leg_year = [
        Line2D(
            [0],
            [0],
            color="#555555",
            lw=YEAR_LINEWIDTH[y],
            linestyle="-",
            label=f"Year ratio level = {y}",
        )
        for y in sorted(YEAR_LINEWIDTH)
    ]
    fig.legend(
        handles=leg_hour + leg_day + leg_year,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=9,
        frameon=True,
        framealpha=0.92,
    )

    n_fronts = len(by_panel["baseline"]) + len(by_panel["flexible"])
    ttl = title or (
        f"Tariff sensitivity — Pareto overlays ({drought})\n"
        f"Colors (tab10-style) = peak hours; line style = day ratio; width = year ratio; "
        f"flexible panel uses muted colors ({n_fronts} fronts)"
    )
    fig.suptitle(ttl, fontsize=11, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.22, wspace=0.12)

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
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    paths = discover_csvs(args.drought, args.pareto_dir)
    if not paths:
        print(
            f"No Pareto CSVs found under {args.pareto_dir!r} for drought {args.drought!r}. "
            "Run: python3 run_tariff_sensitivity_batch.py --drought ... --cases all"
        )
        return 1

    plot_overlay(args.drought, paths, args.out, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
