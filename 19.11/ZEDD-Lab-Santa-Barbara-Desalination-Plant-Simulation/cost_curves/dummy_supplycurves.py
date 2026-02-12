import pandas as pd
import numpy as np

# %%
def create_water_production(ro_capacity):
    """
    Returns a monthly water production profile based on the RO capacity.
    """
    return np.linspace(0, ro_capacity/12, 20)


def create_flat_elec(ro_capacity, water_prod):
    """
    Returns a constant cost profile scaled based on RO capacity
    """
    return np.ones_like(water_prod) * ro_capacity * 800

def create_peaky_elec(ro_capacity, water_prod):
    """
    Returns a electricity cost curve that's sectioned into a lower and higher slope piece-wise linear curve
    """
    cost_of_desal = {"3125":2040,
            "4250":1875,
            "5055":1625,
            "7500":1510,
            "10000":1405}[str(ro_capacity)] # USD /AF

    idx = int(0.8 * len(water_prod))  # neck of the curve happens at 80% capacity (assumed)

    neck_cost = water_prod[idx] * cost_of_desal * 0.5
    max_cost = np.max(water_prod) * cost_of_desal * 1.1
    first_sec = np.linspace(0, neck_cost, idx)
    second_sec = np.linspace(neck_cost, max_cost, len(water_prod) - idx)
    return np.concatenate((first_sec, second_sec))

def create_neg_elec(ro_capacity, water_prod):
    """
    Returns a quadratic cost curve that has a negative portion during low production
    """
    a = 0.2
    b = -ro_capacity * 0.1
    c = ro_capacity*2

    annual_prod = 12 * water_prod

    return a * annual_prod**2 + b * annual_prod + c


def create_capital_cost(ro_capacity):
    """
    Returns the capital cost of the RO plant based on its annual capacity.
    """
    return {"3125":0,
            "4250":750000,
            "5055":15e6,
            "7500":30e6,
            "10000":45e6}[str(ro_capacity)]


def create_df(ro_capacity, scenario, summer_winter_split=1, casenum=1):

    water_prod = create_water_production(ro_capacity)
    if scenario == "flat":
        elec_price = create_flat_elec(ro_capacity, water_prod)
    elif scenario == "peaky":
        elec_price = create_peaky_elec(ro_capacity, water_prod)
    elif scenario == "neg":
        elec_price = create_neg_elec(ro_capacity, water_prod)
    
    capital_cost = create_capital_cost(ro_capacity)
    fixed_cost = 366000
    labor_cost = 400000

    df_operating_sum = pd.DataFrame({
        "water_production_AF_month": water_prod,
        "electricity_cost_usd_month": elec_price,
        "fixed_cost_usd_month": fixed_cost,
    })
    df_operating_win = pd.DataFrame({
        "water_production_AF_month": water_prod,
        "electricity_cost_usd_month": elec_price*summer_winter_split,
        "fixed_cost_usd_month": fixed_cost,
    })
    df_capital = pd.DataFrame({
        "capital_upgrade_cost_usd": [capital_cost],
        "labor_cost_usd_month": [labor_cost]
    })

    df_operating_sum.to_csv(f"dummy_results/{casenum}_summer.csv", index=False)
    df_operating_win.to_csv(f"dummy_results/{casenum}_winter.csv", index=False)
    df_capital.to_csv(f"dummy_results/{casenum}_overall.csv", index=False)

# %%
np.random.seed(42)
capacities = [3125, 4250, 5055, 7500, 10000]
scenarios = ["flat", "peaky", "neg"]
sum_win_peaks = [1, 0.5, 0.1]  # summer/winter split for each capacity

# %%
casenum = 1  # Case number for the output files
metadata_df = pd.DataFrame({
    "case_number": [],
    "ro_capacity": [],
    "scenario": [],
    "summer_winter_split": []
})
for c in capacities:
    for s in scenarios:
        for sw in sum_win_peaks:
            create_df(ro_capacity=c, scenario=s, summer_winter_split=sw, casenum=casenum)
            tmp={
                "case_number": [int(casenum)],
                "ro_capacity": [c],
                "scenario": [s],
                "summer_winter_split": [sw]
            }
            metadata_df = pd.concat([metadata_df, pd.DataFrame(tmp)])
            casenum += 1



# %%
metadata_df.to_csv("dummy_results/metadata.csv", index=False)


