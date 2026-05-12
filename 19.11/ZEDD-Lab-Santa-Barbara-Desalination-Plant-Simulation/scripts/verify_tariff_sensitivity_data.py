#!/usr/bin/env python3
"""
Quality checks for cost_curves/supply_curve_tariff_sensitivity/ (stdlib only).

- Inventory hour*_day*_year* folders and baseline|flexible CSV pairs
- Confirm uniform row counts and presence of summer/winter files
- Count *_overall.csv (required by CostCurveLoader today)
- Summarize crude "annual electricity" proxies: 6 * summer_elec + 6 * winter_elec
  at max production, idle, and mean over curve rows (same index pairing)

Run from project root:
  python3 scripts/verify_tariff_sensitivity_data.py
  python3 scripts/verify_tariff_sensitivity_data.py --case 3mpd_30vessels
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple


def _elec_key(header: List[str]) -> str:
    for k in header:
        if "electricity" in k.lower():
            return k
    raise ValueError(f"No electricity column in {header}")


def _water_key(header: List[str]) -> str:
    for k in header:
        if "water" in k.lower() and "production" in k.lower():
            return k
    raise ValueError(f"No water production column in {header}")


def read_curve(path: str) -> Tuple[List[float], List[float]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return [], []
    wk, ek = _water_key(r.fieldnames or []), _elec_key(r.fieldnames or [])
    wv = [float(row[wk]) for row in rows]
    ev = [float(row[ek]) for row in rows]
    return wv, ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="3mpd_30vessels", help="Case stem for CSV filenames")
    ap.add_argument(
        "--root",
        default=os.path.join("cost_curves", "supply_curve_tariff_sensitivity"),
        help="Path to supply_curve_tariff_sensitivity directory",
    )
    args = ap.parse_args()

    root = args.root
    case = args.case
    if not os.path.isdir(root):
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    pat = re.compile(r"^hour(\d+)_day(\d+)_year(\d+)$")
    folders = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    parsed: List[Tuple[str, int, int, int]] = []
    odd: List[str] = []
    for f in folders:
        m = pat.match(f)
        if m:
            parsed.append((f, int(m.group(1)), int(m.group(2)), int(m.group(3))))
        else:
            odd.append(f)

    overall_n = 0
    for dirpath, _, filenames in os.walk(root):
        overall_n += sum(1 for fn in filenames if fn.endswith("_overall.csv"))

    rows_out: List[dict] = []
    bad: List[str] = []

    for folder, _h, _d, _y in parsed:
        for sub in ("baseline", "flexible"):
            sp = os.path.join(root, folder, sub, f"{case}_summer.csv")
            wp = os.path.join(root, folder, sub, f"{case}_winter.csv")
            if not (os.path.isfile(sp) and os.path.isfile(wp)):
                bad.append(f"missing pair: {folder}/{sub}")
                continue
            sw, se = read_curve(sp)
            ww, we = read_curve(wp)
            if len(sw) != len(we) or len(se) != len(we):
                bad.append(f"length mismatch: {folder}/{sub}")
                continue
            n = len(se)
            annual_max = 6 * se[-1] + 6 * we[-1]
            annual_idle = 6 * se[0] + 6 * we[0]
            mid = n // 2
            annual_mid = 6 * se[mid] + 6 * we[mid]
            annual_mean = sum(6 * se[i] + 6 * we[i] for i in range(n)) / n
            # row-wise water mismatch (informational)
            max_dw = max(abs(sw[i] - ww[i]) for i in range(n))
            rows_out.append(
                {
                    "folder": folder,
                    "sub": sub,
                    "n": n,
                    "annual_max_prod_usd": annual_max,
                    "annual_idle_usd": annual_idle,
                    "annual_mid_idx_usd": annual_mid,
                    "annual_curve_mean_usd": annual_mean,
                    "max_abs_water_diff_summer_winter_af": max_dw,
                }
            )

    print("=== supply_curve_tariff_sensitivity checks ===")
    print(f"Root: {os.path.abspath(root)}")
    print(f"Case: {case}")
    print(f"hour_day_year folders: {len(parsed)}")
    if odd:
        print(f"Other subdirs (not hour_day_year): {odd}")
    print(f"Expected CSV pairs (x2 baseline|flexible): {len(parsed) * 2}")
    print(f"Loaded OK: {len(rows_out)}")
    print(f"_overall.csv files (any depth): {overall_n}")
    if bad:
        print(f"Issues ({len(bad)}):")
        for b in bad[:30]:
            print(" ", b)
        if len(bad) > 30:
            print("  ...")
    if len({r['n'] for r in rows_out}) == 1:
        print(f"Uniform curve rows per file: {list({r['n'] for r in rows_out})[0]}")
    else:
        print("Row counts:", {r["n"] for r in rows_out})

    def summarize(sub: str, key: str) -> None:
        xs = [r[key] for r in rows_out if r["sub"] == sub]
        if not xs:
            return
        lo, hi = min(xs), max(xs)
        mu = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        cv = 100 * sd / mu if mu else 0.0
        rng = 100 * (hi - lo) / mu if mu else 0.0
        print(f"\n[{sub}] {key}")
        print(f"  n={len(xs)} min={lo:,.0f} max={hi:,.0f} mean={mu:,.0f}")
        print(f"  std={sd:,.0f}  CV%={cv:.2f}  (max-min)/mean%={rng:.2f}")

    print("\n--- Crude annual electricity USD proxies (6*summer + 6*winter same point all year) ---")
    for sub in ("baseline", "flexible"):
        for key in (
            "annual_max_prod_usd",
            "annual_idle_usd",
            "annual_curve_mean_usd",
        ):
            summarize(sub, key)

    # summer/winter water grid alignment
    mx = max(r["max_abs_water_diff_summer_winter_af"] for r in rows_out) if rows_out else 0
    print("\n--- Summer vs winter water_production row pairing ---")
    print(f"  max |ΔAF| across rows and scenarios: {mx:.6f} AF/month")

    base = {r["folder"]: r["annual_max_prod_usd"] for r in rows_out if r["sub"] == "baseline"}
    flex = {r["folder"]: r["annual_max_prod_usd"] for r in rows_out if r["sub"] == "flexible"}
    common = sorted(set(base) & set(flex))
    if common:
        diffs = [100 * (flex[k] - base[k]) / base[k] for k in common]
        print("\n--- Flexible vs baseline (annual_max_prod proxy) ---")
        print(
            f"  pct diff flex vs base: min={min(diffs):.2f}% "
            f"max={max(diffs):.2f}% mean={statistics.mean(diffs):.2f}%"
        )

    print("\n--- CostCurveLoader integration (manual) ---")
    print("  Today: requires *_overall.csv + non-recursive subfolder scan.")
    print("  This tree: nested hour_day_year/baseline|flexible and no _overall.csv → not auto-discovered.")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
