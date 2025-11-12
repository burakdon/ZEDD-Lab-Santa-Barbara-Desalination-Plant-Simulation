#!/usr/bin/env python3
"""
Overlay saved Pareto fronts for multiple cases without rerunning simulations.

Usage examples:
  python overlay_pareto.py --drought pers87_sev0.83n_4 --cases 14 43 \
      --labels "Case 14 (peaky)" "Case 43 (neg)" \
      --out result/plots/pareto/overlay_pers87_14_vs_43.png

If labels are omitted, labels default to case numbers.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_pareto_csv(drought: str, case_id):
    path = f"result/data/pareto/pareto_{drought}_case_{case_id}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pareto CSV not found: {path}. Run main.py for this case first.")
    df = pd.read_csv(path)
    return df


def overlay(drought: str, cases: list, labels: list = None, out: str = None, title: str = None):
    plt.figure(figsize=(7, 5))
    labs = labels if labels and len(labels) == len(cases) else [f"case {c}" for c in cases]

    for case, lab in zip(cases, labs):
        df = load_pareto_csv(drought, case)
        plt.scatter(df['cost'], df['risk_months_supply'], s=22, label=lab)

    plt.xlabel('cost')
    plt.ylabel('# demand months left in storage (risk)')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"Saved overlay plot to: {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--drought', required=True, help='Drought type string used in saved filenames')
    p.add_argument('--cases', nargs='+', required=True, help='List of case identifiers to overlay')
    p.add_argument('--labels', nargs='*', help='Optional list of labels same length as cases')
    p.add_argument('--out', help='Output PNG path for the overlay plot')
    p.add_argument('--title', help='Custom title for the plot')
    args = p.parse_args()

    cases = [c.strip() for c in args.cases if c.strip()]
    labels = args.labels
    if labels:
        labels = [lab.strip() for lab in labels]
    overlay(args.drought, cases, labels, args.out, args.title)


if __name__ == '__main__':
    main()


