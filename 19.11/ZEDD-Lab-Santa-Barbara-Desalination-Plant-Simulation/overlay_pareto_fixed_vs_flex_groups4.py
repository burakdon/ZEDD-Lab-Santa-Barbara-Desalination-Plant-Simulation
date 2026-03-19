#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import overlay_pareto_fixed_vs_flex as ov


def plot_group(
    drought: str,
    curve_name: str,
    fixed_summary: str,
    out_path: str,
    risk_stat: str,
    cost_stat: str,
    fixed_fractions: Optional[List[float]],
    connect_fixed: bool,
    quiet: bool,
) -> None:
    baseline_case = f"basetariff_baseline/{curve_name}"
    flexible_case = f"basetariff_flexible/{curve_name}"

    plt.figure(figsize=(7, 5))

    for case_id in [baseline_case, flexible_case]:
        color, is_baseline = ov.get_color_for_curve(case_id)

        flex = ov.load_flex_pareto_csv(drought, case_id)

        # Flex points
        if is_baseline:
            plt.scatter(
                flex["cost"], flex["risk_months_supply"],
                s=22, c=color, marker="o", alpha=0.65,
                edgecolors="black", linewidths=0.5,
                label=f"{case_id} (flex)",
            )
        else:
            plt.scatter(
                flex["cost"], flex["risk_months_supply"],
                s=22, c=color, marker="o", alpha=0.65,
                edgecolors=color, linewidths=1.3, facecolors="none",
                label=f"{case_id} (flex)",
            )

        # Fixed points
        fixed = ov.load_fixed_points(
            summary_csv=fixed_summary,
            case_id=case_id,
            fixed_fractions=fixed_fractions,
            risk_stat=risk_stat,
            cost_stat=cost_stat,
        )

        if connect_fixed and len(fixed) > 1:
            plt.plot(
                fixed["cost"], fixed["risk_months_supply"],
                linewidth=1.0, alpha=0.35, color=color
            )

        if is_baseline:
            plt.scatter(
                fixed["cost"], fixed["risk_months_supply"],
                s=90, c=color, marker="*",
                alpha=0.9, edgecolors="black", linewidths=0.6,
                label=f"{case_id} (fixed)",
            )
        else:
            plt.scatter(
                fixed["cost"], fixed["risk_months_supply"],
                s=90, c=color, marker="*",
                alpha=0.9, edgecolors=color, linewidths=1.2,
                facecolors="none",
                label=f"{case_id} (fixed)",
            )

        if not quiet:
            print(f"[INFO] {curve_name}: {case_id} flex points={len(flex)}, fixed points={len(fixed)}")

    plt.xlabel("cost")
    plt.ylabel("# demand months left in storage (risk)")
    plt.title(f"Flex vs Fixed — {curve_name}")
    plt.legend(fontsize=8, frameon=True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=175, bbox_inches="tight")
    plt.close()
    if not quiet:
        print(f"[OK] Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drought", required=True, help="Drought string used in saved filenames.")
    p.add_argument("--fixed-summary", default=None, help="Path to fixed summary CSV (recommended).")
    p.add_argument("--risk-stat", default="mean", choices=["mean", "p10", "p50", "p90"])
    p.add_argument("--cost-stat", default="mean", choices=["mean", "p10", "p50", "p90"])
    p.add_argument("--fixed-fractions", nargs="*", default=None, help="Optional fractions, e.g. 1.0 0.8 0.6")
    p.add_argument("--outdir", default="result/plots/pareto", help="Output directory for group plots.")
    p.add_argument("--curves", nargs="*", default=None, help="Optional list of curve_names to plot.")
    p.add_argument("--no-connect-fixed", action="store_true", help="Do not connect fixed points with a line.")
    p.add_argument("--quiet", action="store_true", help="Suppress info prints.")
    args = p.parse_args()

    fixed_summary_path = args.fixed_summary
    if fixed_summary_path is None:
        # Will auto-discover but may error if multiple exist; passing --fixed-summary avoids that.
        fixed_summary_path = ov.find_fixed_summary_csv(explicit_path=None, drought=args.drought)

    fixed_fractions = None
    if args.fixed_fractions:
        fixed_fractions = [float(x) for x in args.fixed_fractions]

    curve_names = args.curves if args.curves else list(ov.CURVE_COLORS.keys())

    for curve_name in curve_names:
        out_path = os.path.join(
            args.outdir,
            f"overlay_flex_vs_fixed_group4_{curve_name}_{args.drought}.png"
        )
        plot_group(
            drought=args.drought,
            curve_name=curve_name,
            fixed_summary=fixed_summary_path,
            out_path=out_path,
            risk_stat=args.risk_stat,
            cost_stat=args.cost_stat,
            fixed_fractions=fixed_fractions,
            connect_fixed=(not args.no_connect_fixed),
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()