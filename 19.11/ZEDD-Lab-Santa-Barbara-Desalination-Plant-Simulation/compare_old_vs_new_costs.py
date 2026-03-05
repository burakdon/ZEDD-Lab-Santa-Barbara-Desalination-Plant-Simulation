import os
import pandas as pd
import matplotlib.pyplot as plt

OLD_DIR = "cost_curves/old_data"
NEW_DIR = "cost_curves/new_data"

# OPTION A: specify cases manually
# CASES = ["3mpd_30vessels", "4mpd_36vessels"]

# OPTION B: uncomment this to run ALL cases found in old_data/basetariff_baseline
CASES = sorted({
    fn.replace("_summer.csv", "")
    for fn in os.listdir(os.path.join(OLD_DIR, "basetariff_baseline"))
    if fn.endswith("_summer.csv")
})

FOLDERS = ["basetariff_baseline", "basetariff_flexible"]

def load_case(root, folder, case):
    base = os.path.join(root, folder)
    overall = pd.read_csv(os.path.join(base, f"{case}_overall.csv"))
    summer = pd.read_csv(os.path.join(base, f"{case}_summer.csv"))
    winter = pd.read_csv(os.path.join(base, f"{case}_winter.csv"))
    return overall, summer, winter

for case in CASES:
    data = {}
    for folder in FOLDERS:
        old_overall, old_summer, old_winter = load_case(OLD_DIR, folder, case)
        new_overall, new_summer, new_winter = load_case(NEW_DIR, folder, case)
        data[folder] = {
            "old_overall": old_overall,
            "new_overall": new_overall,
            "old_summer": old_summer,
            "new_summer": new_summer,
            "old_winter": old_winter,
            "new_winter": new_winter,
        }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle(f"Old vs new cost curves — {case}")

    # Baseline summer
    ax = axes[0, 0]
    ds = data["basetariff_baseline"]
    ax.plot(ds["old_summer"]["water_production_AF_month"],
            ds["old_summer"]["electricity_cost_usd_month"] + ds["old_summer"]["fixed_cost_usd_month"],
            label="old", color="C0")
    ax.plot(ds["new_summer"]["water_production_AF_month"],
            ds["new_summer"]["electricity_cost_usd_month"] + ds["new_summer"]["fixed_cost_usd_month"],
            label="new", color="C1")
    ax.set_title("Baseline — summer")
    ax.set_ylabel("op cost (USD/month)")
    ax.legend()

    # Baseline winter
    ax = axes[1, 0]
    ax.plot(ds["old_winter"]["water_production_AF_month"],
            ds["old_winter"]["electricity_cost_usd_month"] + ds["old_winter"]["fixed_cost_usd_month"],
            label="old", color="C0")
    ax.plot(ds["new_winter"]["water_production_AF_month"],
            ds["new_winter"]["electricity_cost_usd_month"] + ds["new_winter"]["fixed_cost_usd_month"],
            label="new", color="C1")
    ax.set_title("Baseline — winter")
    ax.set_ylabel("op cost (USD/month)")
    ax.set_xlabel("AF/month")

    # Flexible summer
    ax = axes[0, 1]
    ds = data["basetariff_flexible"]
    ax.plot(ds["old_summer"]["water_production_AF_month"],
            ds["old_summer"]["electricity_cost_usd_month"] + ds["old_summer"]["fixed_cost_usd_month"],
            label="old", color="C0")
    ax.plot(ds["new_summer"]["water_production_AF_month"],
            ds["new_summer"]["electricity_cost_usd_month"] + ds["new_summer"]["fixed_cost_usd_month"],
            label="new", color="C1")
    ax.set_title("Flexible — summer")

    # Flexible winter
    ax = axes[1, 1]
    ax.plot(ds["old_winter"]["water_production_AF_month"],
            ds["old_winter"]["electricity_cost_usd_month"] + ds["old_winter"]["fixed_cost_usd_month"],
            label="old", color="C0")
    ax.plot(ds["new_winter"]["water_production_AF_month"],
            ds["new_winter"]["electricity_cost_usd_month"] + ds["new_winter"]["fixed_cost_usd_month"],
            label="new", color="C1")
    ax.set_title("Flexible — winter")
    ax.set_xlabel("AF/month")

    # Shared capital cost panel
    ax = axes[0, 2]
    labels = ["base old", "base new", "flex old", "flex new"]
    heights = []
    for folder in FOLDERS:
        ds = data[folder]
        heights.append(float(ds["old_overall"]["capital_upgrade_cost_usd"].iloc[0]))
        heights.append(float(ds["new_overall"]["capital_upgrade_cost_usd"].iloc[0]))
    ax.bar(labels, heights, alpha=0.7)
    ax.set_title("Capital upgrade cost")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=30)

    axes[1, 2].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"result/plots/pareto/costcurve_old_vs_new_{case}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved", out)