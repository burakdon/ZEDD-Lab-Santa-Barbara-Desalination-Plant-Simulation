#!/usr/bin/env python3
"""
overlay_pareto_fixed_vs_flex.py

Overlay flex-optimization Pareto fronts with fixed-fraction outcomes.

Flex Pareto points are read from:
  result/data/pareto/pareto_{drought}_case_{CASE}.csv
with a fallback to:
  result/data/pareto/pareto_{drought}_case_{RAW_CASE_ID}.csv

Fixed points are read from a summary CSV produced by the fixed experiment
(e.g., result/fixed_desal/summary.csv). Fixed points are plotted with a '*'
marker, matching the flex case color.

Marker conventions (mirrors the existing overlay_pareto.py style):
  - baseline cases: filled circles for flex; filled stars for fixed (black edge)
  - flexible cases: hollow circles for flex; hollow stars for fixed (colored edge)

Notes on fixed summary schema:
- This script supports columns like:
    risk_months_mean, risk_months_p10, risk_months_p50, risk_months_p90
  and similarly for cost (cost_mean, ...).
- By default it uses the mean columns when available.

Example:
  conda activate spyder-env
  PYTHONPATH=src python overlay_pareto_fixed_vs_flex.py \
    --drought pers87_sev0.83n_4 \
    --cases basetariff_baseline/3mpd_30vessels basetariff_flexible/3mpd_30vessels \
    --out result/plots/pareto/overlay_flex_vs_fixed.png
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CURVE_COLORS = {
    "3mpd_30vessels": "#d62728",
    "3mpd_36vessels": "#ff7f0e",
    "4mpd_30vessels": "#2ca02c",
    "4mpd_36vessels": "#1f77b4",
    "6mpd_36vessels": "#9467bd",
    "8mpd_36vessels": "#17becf",
}


def parse_case_identifier(case_identifier: str) -> Tuple[Optional[str], str]:
    case_str = str(case_identifier).strip()
    if "/" in case_str:
        folder, curve_name = case_str.split("/", 1)
        folder_short = folder.split("_")[-1] if "_" in folder else folder
        return folder_short, curve_name
    return None, case_str


def format_case_for_filename(case_identifier: str) -> str:
    folder_short, curve_name = parse_case_identifier(case_identifier)
    if folder_short:
        return f"{folder_short}_{curve_name}"
    return str(case_identifier).strip()


def mute_color(color_hex: str, saturation_reduction: float = 0.5, brightness_increase: float = 0.1) -> str:
    rgb = mcolors.hex2color(color_hex)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] = max(0.0, hsv[1] - saturation_reduction)
    hsv[2] = min(1.0, hsv[2] + brightness_increase)
    rgb_muted = mcolors.hsv_to_rgb(hsv)
    return mcolors.rgb2hex(rgb_muted)


def get_color_for_curve(case_identifier: str) -> Tuple[str, bool]:
    folder_short, curve_name = parse_case_identifier(case_identifier)
    base_color = CURVE_COLORS.get(curve_name, "#808080")
    is_baseline = (folder_short is not None) and ("baseline" in folder_short.lower())
    if is_baseline:
        return base_color, True
    return mute_color(base_color), False


def _candidate_flex_paths(drought: str, case_id: str, base_dir: str = "result/data/pareto") -> List[str]:
    case_filename = format_case_for_filename(case_id)
    return [
        os.path.join(base_dir, f"pareto_{drought}_case_{case_filename}.csv"),
        os.path.join(base_dir, f"pareto_{drought}_case_{case_id}.csv"),
    ]


def load_flex_from_summary(summary_csv: str, case_id: str, cost_stat: str = "mean", risk_stat: str = "mean") -> pd.DataFrame:
    """
    Load flexible Pareto points from a summary.csv file.
    Filters by case_id and extracts all points (ignoring fraction if present).
    """
    df = pd.read_csv(summary_csv)
    
    # Find case column
    case_col = None
    for c in ["case", "case_id", "case_identifier", "case_name", "case_label", "curve", "curve_id"]:
        if c in df.columns:
            case_col = c
            break
    if case_col is None:
        raise KeyError(f"Flex summary {summary_csv} missing case id column. Found: {list(df.columns)}")
    
    # Find cost and risk columns
    stat = risk_stat.lower().strip()
    cstat = cost_stat.lower().strip()
    
    cost_candidates = [
        f"cost_{cstat}",
        f"Jcost_{cstat}",
        "cost_mean",
        "Jcost_mean",
        "cost",
        "Jcost",
        "total_cost_mean",
        "total_cost",
    ]
    risk_candidates = [
        f"risk_months_{stat}",
        f"risk_months_supply_{stat}",
        f"risk_{stat}",
        "risk_months_mean",
        "risk_months_supply_mean",
        "risk_mean",
        "risk_months_supply",
        "risk",
        "risk_months",
        "Jrisk_mean",
        "Jrisk",
    ]
    
    cost_col = _pick_col(df, cost_candidates, "cost")
    risk_col = _pick_col(df, risk_candidates, "risk")
    
    # Filter for this case - try exact match first
    sel = df[df[case_col].astype(str) == str(case_id)]
    if sel.empty:
        alt = format_case_for_filename(case_id)
        sel = df[df[case_col].astype(str) == str(alt)]
    if sel.empty:
        # Try matching by case_label if it contains the curve name
        folder_short, curve_name = parse_case_identifier(case_id)
        sel = df[df[case_col].astype(str).str.contains(curve_name, regex=False)]
        if folder_short and not sel.empty:
            # Further filter by folder (baseline/flexible)
            sel2 = sel[sel[case_col].astype(str).str.contains(folder_short, regex=False)]
            if not sel2.empty:
                sel = sel2
    
    if sel.empty:
        uniq = df[case_col].astype(str).unique()
        sample = ", ".join(sorted(uniq)[:20])
        raise ValueError(
            f"No flex results found for case '{case_id}' in {summary_csv}. "
            f"Example case values: {sample}"
        )
    
    # If fraction column exists, filter out rows with fraction values
    # (flexible optimization results should not have fraction, or have fraction=None/NaN)
    if "fraction" in sel.columns:
        # Keep rows where fraction is NaN/None (flexible optimization results)
        # or if all rows have fraction, use all of them (backward compatibility)
        if sel["fraction"].isna().any():
            sel = sel[sel["fraction"].isna()]
        # If no NaN fractions, assume all rows are flexible optimization results
        # (the caller should filter fixed fraction results separately)
    
    # Extract cost and risk columns (ignore fraction if present)
    out = sel[[cost_col, risk_col]].copy()
    out = out.rename(columns={cost_col: "cost", risk_col: "risk_months_supply"})
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["risk_months_supply"] = pd.to_numeric(out["risk_months_supply"], errors="coerce")
    return out.dropna(subset=["cost", "risk_months_supply"])


def load_flex_pareto_csv(drought: str, case_id: str, flex_dir: str = None, flex_summary: str = None, cost_stat: str = "mean", risk_stat: str = "mean") -> pd.DataFrame:
    """
    Load flexible Pareto points from either:
    1. A summary.csv file (if flex_summary is provided)
    2. Traditional Pareto CSV files (if flex_dir is provided or as fallback)
    """
    if flex_summary:
        return load_flex_from_summary(flex_summary, case_id, cost_stat, risk_stat)
    
    # Fall back to traditional Pareto CSV files
    if flex_dir is None:
        flex_dir = "result/data/pareto"
    
    tried = _candidate_flex_paths(drought, case_id, base_dir=flex_dir)
    path = None
    for pth in tried:
        if os.path.exists(pth):
            path = pth
            break
    if path is None:
        raise FileNotFoundError(
            "Flex Pareto CSV not found for case '{}' in directory '{}'. Tried:\n  - {}".format(
                case_id, flex_dir, "\n  - ".join(tried)
            )
        )

    df = pd.read_csv(path)

    if "cost" not in df.columns:
        for alt in ["Jcost", "Jcost_mean", "cost_mean", "total_cost", "total_cost_mean"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "cost"})
                break

    if "risk_months_supply" not in df.columns:
        for alt in ["Jrisk", "Jrisk_mean", "risk", "risk_mean", "risk_months", "risk_months_supply_mean"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "risk_months_supply"})
                break

    if ("cost" not in df.columns) or ("risk_months_supply" not in df.columns):
        raise ValueError(f"Flex Pareto CSV {path} missing required columns. Found: {list(df.columns)}")

    df = df[["cost", "risk_months_supply"]].copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["risk_months_supply"] = pd.to_numeric(df["risk_months_supply"], errors="coerce")
    return df.dropna(subset=["cost", "risk_months_supply"])


def find_fixed_summary_csv(explicit_path: Optional[str], drought: Optional[str]) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--fixed-summary not found: {explicit_path}")
        return explicit_path

    candidates = [
        os.path.join("result", "fixed_desal", "summary.csv"),
        os.path.join("result", "fixed_desal", f"summary_{drought}.csv") if drought else None,
        os.path.join("result", "fixed_desal_experiment_output", "summary.csv"),
        os.path.join("result", "fixed_desal_experiment_output", f"summary_{drought}.csv") if drought else None,
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c

    hits = glob.glob(os.path.join("result", "**", "summary.csv"), recursive=True)
    hits = [h for h in hits if os.path.isfile(h)]
    if len(hits) == 1:
        return hits[0]

    fixed_hits = [h for h in hits if "fixed" in h.lower()]
    if len(fixed_hits) == 1:
        return fixed_hits[0]

    if hits:
        msg = "Could not uniquely infer fixed summary CSV. Found multiple:\n  - " + "\n  - ".join(sorted(hits)[:50])
        msg += "\nPass --fixed-summary <path> to choose."
        raise FileNotFoundError(msg)

    raise FileNotFoundError("Could not find a fixed summary CSV. Pass --fixed-summary <path>.")


def _pick_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Fixed summary missing {label} column. Looked for {candidates}. Found {list(df.columns)}"
    )


def load_fixed_points(
    summary_csv: str,
    case_id: str,
    fixed_fractions: Optional[List[float]] = None,
    risk_stat: str = "mean",
    cost_stat: str = "mean",
) -> pd.DataFrame:
    """
    Load fixed points for a given case, returning DataFrame with columns:
      cost, risk_months_supply, fraction

    risk_stat/cost_stat can be: 'mean', 'p10', 'p50', 'p90'
    """
    df = pd.read_csv(summary_csv)

    case_col = None
    for c in ["case", "case_id", "case_identifier", "case_name", "case_label", "curve", "curve_id"]:
        if c in df.columns:
            case_col = c
            break
    if case_col is None:
        raise KeyError(f"Fixed summary {summary_csv} missing case id column. Found: {list(df.columns)}")

    frac_col = None
    for c in ["fraction", "desal_fraction", "fixed_fraction", "frac"]:
        if c in df.columns:
            frac_col = c
            break
    if frac_col is None:
        raise KeyError(f"Fixed summary {summary_csv} missing fraction column. Found: {list(df.columns)}")

    stat = risk_stat.lower().strip()
    if stat not in {"mean", "p10", "p50", "p90"}:
        raise ValueError(f"risk_stat must be one of mean,p10,p50,p90; got {risk_stat}")
    cstat = cost_stat.lower().strip()
    if cstat not in {"mean", "p10", "p50", "p90"}:
        raise ValueError(f"cost_stat must be one of mean,p10,p50,p90; got {cost_stat}")

    cost_candidates = [
        f"cost_{cstat}",
        f"Jcost_{cstat}",
        "cost_mean",
        "Jcost_mean",
        "cost",
        "Jcost",
        "total_cost_mean",
        "total_cost",
    ]
    risk_candidates = [
        f"risk_months_{stat}",
        f"risk_months_supply_{stat}",
        f"risk_{stat}",
        "risk_months_mean",
        "risk_months_supply_mean",
        "risk_mean",
        "risk_months_supply",
        "risk",
        "risk_months",
        "Jrisk_mean",
        "Jrisk",
    ]

    cost_col = _pick_col(df, cost_candidates, "cost")
    risk_col = _pick_col(df, risk_candidates, "risk")

    sel = df[df[case_col].astype(str) == str(case_id)]
    if sel.empty:
        alt = format_case_for_filename(case_id)
        sel = df[df[case_col].astype(str) == str(alt)]
    if sel.empty:
        folder_short, curve_name = parse_case_identifier(case_id)
        sel = df[df[case_col].astype(str).str.contains(curve_name, regex=False)]
        if folder_short and not sel.empty:
            sel2 = sel[sel[case_col].astype(str).str.contains(folder_short, regex=False)]
            if not sel2.empty:
                sel = sel2

    if sel.empty:
        uniq = df[case_col].astype(str).unique()
        sample = ", ".join(sorted(uniq)[:20])
        raise ValueError(
            f"No fixed results found for case '{case_id}' in {summary_csv} (case column '{case_col}'). "
            f"Example case values: {sample}"
        )

    out = sel[[cost_col, risk_col, frac_col]].copy()
    out = out.rename(columns={cost_col: "cost", risk_col: "risk_months_supply", frac_col: "fraction"})
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    out["risk_months_supply"] = pd.to_numeric(out["risk_months_supply"], errors="coerce")
    out["fraction"] = pd.to_numeric(out["fraction"], errors="coerce")
    out = out.dropna(subset=["cost", "risk_months_supply", "fraction"])

    if fixed_fractions:
        masks = [np.isclose(out["fraction"].values, f, rtol=0.0, atol=1e-6) for f in fixed_fractions]
        mask = np.logical_or.reduce(masks) if masks else np.ones(len(out), dtype=bool)
        out = out.loc[mask]

    return out.sort_values("fraction", ascending=False)


def overlay_fixed_vs_flex(
    drought: str,
    cases: List[str],
    fixed_summary: Optional[str],
    out: Optional[str],
    title: Optional[str],
    fixed_fractions: Optional[List[float]],
    risk_stat: str = "mean",
    cost_stat: str = "mean",
    connect_fixed: bool = True,
    quiet: bool = False,
    flex_dir: str = None,
    flex_summary: str = None,
):
    fixed_summary_path = find_fixed_summary_csv(fixed_summary, drought)
    if not quiet:
        print(f"[INFO] Fixed summary: {fixed_summary_path}")
        if flex_summary:
            print(f"[INFO] Flexible results from summary: {flex_summary}")
        elif flex_dir:
            print(f"[INFO] Flexible results directory: {flex_dir}")
        else:
            print(f"[INFO] Flexible results directory: result/data/pareto (default)")

    plt.figure(figsize=(8, 6))

    for case in cases:
        color, is_baseline = get_color_for_curve(case)

        flex = load_flex_pareto_csv(drought, case, flex_dir=flex_dir, flex_summary=flex_summary, cost_stat=cost_stat, risk_stat=risk_stat)
        if not quiet:
            print(f"[INFO] Flex points for {case}: {len(flex)}")

        if is_baseline:
            plt.scatter(
                flex["cost"], flex["risk_months_supply"],
                s=22, c=color, marker="o", alpha=0.65,
                edgecolors="black", linewidths=0.5,
                label=f"{case} (flex)",
            )
        else:
            plt.scatter(
                flex["cost"], flex["risk_months_supply"],
                s=22, c=color, marker="o", alpha=0.65,
                edgecolors=color, linewidths=1.3, facecolors="none",
                label=f"{case} (flex)",
            )

        fixed = load_fixed_points(
            fixed_summary_path,
            case,
            fixed_fractions=fixed_fractions,
            risk_stat=risk_stat,
            cost_stat=cost_stat,
        )
        if not quiet:
            print(f"[INFO] Fixed points for {case}: {len(fixed)}")

        if connect_fixed and len(fixed) > 1:
            plt.plot(fixed["cost"], fixed["risk_months_supply"], linewidth=1.0, alpha=0.35, color=color)

        if is_baseline:
            plt.scatter(
                fixed["cost"], fixed["risk_months_supply"],
                s=90, c=color, marker="*",
                alpha=0.9, edgecolors="black", linewidths=0.6,
                label=f"{case} (fixed)",
            )
        else:
            plt.scatter(
                fixed["cost"], fixed["risk_months_supply"],
                s=90, c=color, marker="*",
                alpha=0.9, edgecolors=color, linewidths=1.2,
                facecolors="none",
                label=f"{case} (fixed)",
            )

    plt.xlabel("cost")
    plt.ylabel("# demand months left in storage (risk)")
    if title:
        plt.title(title)

    plt.legend(fontsize=7, ncol=1, frameon=True)
    plt.tight_layout()

    if out:
        out_abs = os.path.abspath(out)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        ax = plt.gca()
        if (len(ax.collections) + len(ax.lines)) == 0:
            raise RuntimeError("Nothing was plotted. Check CSV paths and case identifiers.")

        plt.savefig(out_abs, dpi=175, bbox_inches="tight")
        if not quiet:
            print(f"[OK] Saved overlay plot to: {out_abs}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drought", required=True, help="Drought string used in saved filenames.")
    p.add_argument("--cases", nargs="+", required=True, help="Case identifiers to overlay.")
    p.add_argument("--fixed-summary", default=None, help="Path to fixed summary CSV (optional).")
    p.add_argument("--flex-summary", default=None, help="Path to flexible results summary CSV (optional).")
    p.add_argument("--flex-dir", default=None, help="Directory containing flexible Pareto CSV files (alternative to --flex-summary)")
    p.add_argument("--fixed-fractions", nargs="*", default=None, help="Optional fractions to plot, e.g., 1.0 0.8 0.6 0.4 0.2")
    p.add_argument("--risk-stat", default="mean", choices=["mean", "p10", "p50", "p90"], help="Which risk statistic to use from fixed summary (default: mean).")
    p.add_argument("--cost-stat", default="mean", choices=["mean", "p10", "p50", "p90"], help="Which cost statistic to use from fixed summary (default: mean).")
    p.add_argument("--out", default=None, help="Output PNG path.")
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument("--no-connect-fixed", action="store_true", help="Do not connect fixed points with a thin line.")
    p.add_argument("--quiet", action="store_true", help="Suppress info prints.")
    args = p.parse_args()

    fixed_fractions = None
    if args.fixed_fractions:
        fixed_fractions = [float(x) for x in args.fixed_fractions]

    overlay_fixed_vs_flex(
        drought=args.drought,
        cases=[c.strip() for c in args.cases if c.strip()],
        fixed_summary=args.fixed_summary,
        out=args.out,
        title=args.title,
        fixed_fractions=fixed_fractions,
        risk_stat=args.risk_stat,
        cost_stat=args.cost_stat,
        connect_fixed=(not args.no_connect_fixed),
        quiet=args.quiet,
        flex_dir=args.flex_dir,
        flex_summary=args.flex_summary,
    )


if __name__ == "__main__":
    main()
