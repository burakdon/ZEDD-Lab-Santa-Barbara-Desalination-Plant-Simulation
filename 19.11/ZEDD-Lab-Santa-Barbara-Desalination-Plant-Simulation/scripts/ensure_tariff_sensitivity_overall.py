#!/usr/bin/env python3
"""
Write missing ``{stem}_overall.csv`` next to summer/winter tariff-sensitivity curves.

Copies capital/labor from ``cost_curves/new_data/basetariff_baseline`` or
``.../basetariff_flexible`` for the same ``stem`` (e.g. 3mpd_30vessels).

Usage (from project root):
  python3 scripts/ensure_tariff_sensitivity_overall.py
  python3 scripts/ensure_tariff_sensitivity_overall.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.join("cost_curves", "supply_curve_tariff_sensitivity"),
        help="Tariff sensitivity root under project cwd",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: missing directory {root}", file=sys.stderr)
        return 1

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_data = os.path.join(project_root, "cost_curves", "new_data")

    n_written = 0
    n_skip = 0
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith("_summer.csv"):
                continue
            stem = fn[: -len("_summer.csv")]
            overall = os.path.join(dirpath, f"{stem}_overall.csv")
            if os.path.isfile(overall):
                n_skip += 1
                continue
            winter = os.path.join(dirpath, f"{stem}_winter.csv")
            if not os.path.isfile(winter):
                print(f"WARN: summer without winter: {os.path.join(dirpath, fn)}")
                continue
            leaf = os.path.basename(dirpath)
            if leaf == "flexible":
                tmpl_dir = os.path.join(new_data, "basetariff_flexible")
            elif leaf == "baseline":
                tmpl_dir = os.path.join(new_data, "basetariff_baseline")
            else:
                print(f"WARN: unexpected leaf dir {leaf!r} in {dirpath}")
                continue
            tmpl = os.path.join(tmpl_dir, f"{stem}_overall.csv")
            if not os.path.isfile(tmpl):
                print(f"ERROR: no template overall: {tmpl}", file=sys.stderr)
                return 1
            if args.dry_run:
                print(f"would copy -> {overall}")
            else:
                shutil.copy2(tmpl, overall)
            n_written += 1

    print(f"Done. wrote={n_written} already_had_overall={n_skip} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
