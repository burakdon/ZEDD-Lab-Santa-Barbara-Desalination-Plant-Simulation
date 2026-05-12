#!/usr/bin/env python3
"""
Batch NSGA-II runs for ``cost_curves/supply_curve_tariff_sensitivity/**`` cases.

Uses the same outputs as ``run_all_cases.py`` (Pareto PNG/CSV + time-series).

Examples (from project root):
  python3 scripts/ensure_tariff_sensitivity_overall.py
  python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --list-cases
  python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --quick --max-cases 1
  python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --cases all
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cost_curve_loader import CostCurveLoader
from main import OptimizationParameters, run
from run_all_cases import save_outputs_for_case


def _tariff_cases(loader: CostCurveLoader) -> list:
    out = []
    for c in loader.get_available_cases():
        if isinstance(c, str) and c.startswith("supply_curve_tariff_sensitivity/"):
            out.append(c)
    return sorted(out)


def main() -> int:
    os.chdir(PROJECT_ROOT)

    ap = argparse.ArgumentParser(description="Run NSGA-II for tariff-sensitivity cost curves")
    ap.add_argument("--drought", default="pers87_sev0.83n_4")
    ap.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help='Case ids or "all" (default: all tariff sensitivity cases)',
    )
    ap.add_argument("--list-cases", action="store_true")
    ap.add_argument("--timeseries-scenario", type=int, default=0)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Tiny NSGA-II budget for smoke tests (overrides cores/max_gen/npop/nfe)",
    )
    ap.add_argument("--max-cases", type=int, default=0, help="If >0, only first N cases (after sort)")
    args = ap.parse_args()

    loader = CostCurveLoader()
    cases = _tariff_cases(loader)

    if args.list_cases:
        print(f"Tariff sensitivity cases ({len(cases)}):")
        for c in cases:
            print(f"  {c}")
        return 0

    if not cases:
        print(
            "No cases found with prefix supply_curve_tariff_sensitivity/. "
            "Run: python3 scripts/ensure_tariff_sensitivity_overall.py",
            file=sys.stderr,
        )
        return 1

    if args.cases is None or (len(args.cases) == 1 and args.cases[0].lower() == "all"):
        run_cases = cases
    else:
        run_cases = []
        for c in args.cases:
            c = str(c).strip()
            if c in cases:
                run_cases.append(c)
            else:
                print(f"WARN: case not in tariff list, skipping: {c}", file=sys.stderr)

    if args.max_cases > 0:
        run_cases = run_cases[: args.max_cases]

    opt_par = OptimizationParameters()
    if args.quick:
        opt_par.max_gen = 3
        opt_par.npop = 8
        opt_par.cores = min(4, opt_par.cores)
        opt_par.nfe = opt_par.max_gen * opt_par.npop
        opt_par.log_freq = max(1, opt_par.nfe // 3)

    print(
        f"Running {len(run_cases)} case(s); drought={args.drought}; "
        f"quick={args.quick}; nfe={opt_par.nfe}; cores={opt_par.cores}"
    )

    for case_identifier in run_cases:
        print(f"--- {case_identifier} ---")
        results = run(opt_par, case_identifier, args.drought)
        save_outputs_for_case(
            opt_par,
            case_identifier,
            args.drought,
            results,
            timeseries_scenario=args.timeseries_scenario,
        )
        print(f"Completed {case_identifier}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
