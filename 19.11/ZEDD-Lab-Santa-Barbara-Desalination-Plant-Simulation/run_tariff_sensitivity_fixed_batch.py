#!/usr/bin/env python3
"""
Fixed monthly desal utilization (same sweep as ``run_all_cases_fixed.py``) on
**baseline** tariff sensitivity supply curves only — for comparison to flexible
NSGA-II Pareto runs.

Delegates to:
  fixed_desal_experiment/run_all_cases_fixed.py --cases all --tariff-sensitivity-baseline-only

Extra CLI arguments are forwarded (e.g. ``--drought``, ``--fractions``, ``--save-timeseries``).

Example:
  python3 run_tariff_sensitivity_fixed_batch.py
  python3 run_tariff_sensitivity_fixed_batch.py --drought pers87_sev0.83n_4 --save-timeseries
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, "fixed_desal_experiment", "run_all_cases_fixed.py")


def main() -> int:
    cmd = [sys.executable, SCRIPT, "--tariff-sensitivity-baseline-only"]
    rest = sys.argv[1:]
    if "--drought" not in rest:
        cmd.extend(["--drought", "pers87_sev0.83n_4"])
    if "--cases" not in rest:
        cmd.extend(["--cases", "all"])
    cmd.extend(rest)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
