# Santa Barbara Water Supply and Desalination Simulation

This repository is a **fork** of a simulation codebase originally developed by **Professor Marta Zaniolo** at the [**ZEDD Lab** (Zaniolo Lab for Environmental Data and Decisions)](https://zeddlab.weebly.com/), Department of Civil and Environmental Engineering, **Duke University** ([faculty profile](https://cee.duke.edu/faculty/marta-zaniolo)). The model represents the City of **Santa Barbara, California** surface-water portfolio, institutional features of regional supply, and the **Charles E. Meyer Desalination Plant**, driven by real hydrology, demand, and tariff-sensitive desalination **cost curves**.

**Fork maintainer:** Burak Donbekci — see [Contributions](#contributions-on-this-fork) below.

**Repository layout:** The active Python project (entry scripts, `src/`, `data/`, `cost_curves/`, and `result/`) lives under `19.11/ZEDD-Lab-Santa-Barbara-Desalination-Plant-Simulation/`. Unless noted otherwise, shell commands below assume:

```bash
cd 19.11/ZEDD-Lab-Santa-Barbara-Desalination-Plant-Simulation
```

---

## System being modeled

The simulation captures major elements of Santa Barbara’s water supply context, including:

- **Three reservoir / import components:** Lake Cachuma, Gibraltar Reservoir, and State Water Project (SWP) deliveries, each with scenario-specific inflows and operating rules consistent with the original ZEDD Lab formulation.
- **The Charles E. Meyer Desalination Plant:** expansion sizing is represented through discrete **capacity tiers** (**MPD** — million gallons per day, labeled `mpd` in code and file names; often written **MGD** elsewhere in US practice) and the number of reverse-osmosis **vessels**, with **seasonal** operating cost from empirical **cost curve** tables (summer / winter) plus amortized capital where configured.
- **Demand and hydrology:** monthly demand and multi-year hydrologic ensembles (e.g., mission tunnel inflows, drought scenario tags such as `pers87_sev0.83n_4`) supplied as text inputs under `data/`.

The governing code paths are documented in module docstrings (notably `src/simulation.py` and reservoir modules under `src/`).

---

## What the simulation does

For each **cost-curve case** (a desalination expansion / tariff scenario) and **drought ensemble**, the model runs a monthly mass-balance over a long horizon for many parallel hydrologic **replicates** (`nsim` in `OptimizationParameters` in `main.py`).

**Multi-objective optimization (NSGA-II)** is performed with [**Platypus**](https://platypus.readthedocs.io/) (`NSGAII` in `main.py`, problem class `src/sb_problem.py`). Each candidate solution specifies:

1. A **continuous** first decision variable mapped to one of several **discrete desalination expansion tiers** (3–8 MPD and 30–36 vessels in six tiers; see `src/capacity_tiers.py`).
2. Parameters of a **radial basis function (RBF)** policy (`src/policy.py`) that maps normalized storage and calendar state to a **monthly desalination production fraction**.

**Objectives** (see `SB.simulate` in `src/simulation.py`):

- **Cost:** mean across hydrologic replicates of cumulative desalination cost (curve-based energy, fixed, labor, capital amortization as configured) plus a large penalty for unmet demand.
- **Supply risk:** mean across replicates of the **negative 25th percentile** of a monthly **“months of supply”** index derived from combined storage relative to net demand — trading off **lower cost** against **higher storage / lower tail risk**.

NSGA-II returns an approximate **Pareto set**; non-dominated filtering and plotting utilities live in `src/plot_optimization.py`. Saved outputs include Pareto figures and CSVs under `result/plots/pareto/` and `result/data/pareto/`, time-series diagnostics under `result/plots/timeseries/` and `result/data/timeseries/`, and pickled result bundles under `result/`.

---

## Project structure (key paths)

| Path | Role |
|------|------|
| `main.py` | Single-case NSGA-II run; CLI for `--case`, `--drought`, `--list-cases`, `--timeseries-scenario`. |
| `run_all_cases.py` | Batch runner over many cost-curve cases; writes Pareto and time-series artifacts. |
| `run_old_new.sh` | Example driver comparing `old_data` vs `new_data` cost-curve sets via `COST_CURVES_SET`. |
| `src/simulation.py` | Core `SB` simulator: reservoirs, demand, cost curves, objectives. |
| `src/sim_individual.py` | Individual-simulation wrapper used for detailed logs / plotting. |
| `src/sb_problem.py` | Platypus `Problem` binding optimization variables to `SB.simulate`. |
| `src/cost_curve_loader.py` | CSV cost-curve discovery, loading, interpolation, and capital amortization helpers. |
| `src/capacity_tiers.py` | Discrete MPD / vessel expansion tiers mapped from the first optimization variable. |
| `src/plot_optimization.py` | Pareto filtering, Pareto and time-series plots, CSV export helpers. |
| `cost_curves/` | Case directories (`*_summer.csv`, `*_winter.csv`, `*_overall.csv`); optional `metadata.csv`; subsets `new_data/` and `old_data/`. |
| `data/` | Hydrology, demand (`d12_predrought.txt`), and scenario-specific mission tunnel files (`mission_<drought>.txt`), etc. |
| `fixed_desal_experiment/` | Parallel workflow for **fixed** desal utilization fractions vs. flexible RBF policy (`run_all_cases_fixed.py`, `fixed_sb.py`). |
| `overlay_pareto*.py`, `compare_old_vs_new_costs.py` | Analysis / visualization scripts for comparing fronts and tariff assumptions. |
| `result/` | Default output tree for optimization runs (also `result_newdata/`, `result_olddata/` in this fork for archived comparisons). |

**Reinforcement learning prototype** (branch `RL-layer`; see below):

| Path | Role |
|------|------|
| `src/water_env.py` | **Gymnasium** `Env` wrapping the physics/cost logic with a direct monthly action (desal fraction). |
| `src/train_rl.py` | **SAC** (Stable-Baselines3) training for cost-focused vs. safety-penalized agents. |
| `src/visualize_rl.py` | Figures and animation under `result/rl/`. |
| `src/compare_pareto_rl.py` | Overlays RL operating points on NSGA-II Pareto CSVs. |
| `src/requirements_rl.txt` | Extra dependencies for the RL stack. |

---

## Contributions on this fork

This fork extends Prof. Zaniolo’s original simulation with engineering and research-supporting workflow improvements, including:

- **Optimization pipeline:** batch orchestration (`run_all_cases.py`), standardized outputs (Pareto PNG/CSV, time-series PNG/CSV, metadata), and helper shell workflows (`run_old_new.sh`).
- **Code structure and performance:** clearer separation of data loading and simulation (e.g., `CostCurveLoader`), optional **Numba**-accelerated reservoir updates in `simulation.py`, and modular capacity labeling via `capacity_tiers.py`.
- **Cost curve infrastructure:** CSV-based seasonal curves, directory scanning (including nested tariff folders), optional `COST_CURVES_SET` environment variable for `old_data` / `new_data` subsets, and integration of capital amortization and production limits into the simulator.
- **Experiments:** fixed-desal benchmark suite under `fixed_desal_experiment/`, overlay and comparison scripts for Pareto analysis across tariff or data versions.

The **original scientific formulation and authorship** of the Santa Barbara model remain with **Prof. Marta Zaniolo** and the ZEDD Lab; this README’s operational descriptions are written for reproducibility on this fork.

---

## Branch: `RL-layer`

The branch **`RL-layer`** (also on `origin`) adds a **reinforcement learning** layer developed as part of a **graduate course project**:

- Wraps the simulation as a **`gymnasium`** environment (`src/water_env.py`).
- Trains **Soft Actor–Critic (SAC)** agents with **Stable-Baselines3** (`src/train_rl.py`) to learn **adaptive monthly desalination** policies under cost and safety-shaped rewards.
- Includes visualization and **Pareto comparison** tooling (`src/visualize_rl.py`, `src/compare_pareto_rl.py`).

Check out that branch to use RL-specific paths and outputs under `result/rl/`.

---

## Setup

**Requirements:** Python **3** (3.9+ recommended), a C compiler toolchain only if you build **PyTorch** from source (normally wheels suffice).

**1.** Change to the project root (path at the top of this README).

**2.** Install optimization / simulation dependencies (minimum set inferred from `main.py` / `src/` imports):

```bash
pip install "numpy>=1.20" "pandas>=1.3" "matplotlib>=3.5" platypus-opt numba
```

`numba` is optional at import time but **recommended** for performance.

**3. (Optional, RL branch)** Install RL extras:

```bash
pip install -r src/requirements_rl.txt
```

**4.** Parallel runs: `OptimizationParameters` in `main.py` sets `cores` (default **50**). Reduce `cores` to match your machine before long runs to avoid oversubscribing CPUs.

**5.** Cost curve subset: To point `CostCurveLoader` at `cost_curves/old_data` or `cost_curves/new_data` without editing code:

```bash
export COST_CURVES_SET=new_data   # or old_data
```

---

## Usage

**List available cost-curve cases** (identifiers match `*_overall.csv` / folder layout):

```bash
python3 main.py --list-cases
python3 run_all_cases.py --drought pers87_sev0.83n_4 --list-cases
```

**Run NSGA-II for a single case** (default case `4mpd_36vessels` if `--case` omitted):

```bash
python3 main.py --case 4mpd_36vessels --drought pers87_sev0.83n_4 --timeseries-scenario 0
```

**Examples with nested tariff folders** (as used in `run_old_new.sh`):

```bash
python3 main.py --case basetariff_flexible/4mpd_36vessels --drought pers87_sev0.83n_4
```

**Batch many cases:**

```bash
python3 run_all_cases.py --drought pers87_sev0.83n_4 --cases all --timeseries-scenario 0
python3 run_all_cases.py --drought pers87_sev0.83n_4 --cases basetariff_baseline/3mpd_30vessels basetariff_flexible/8mpd_36vessels
```

**Tariff / supply-curve sensitivity** (`cost_curves/supply_curve_tariff_sensitivity/`): ensure capital/labor `*_overall.csv` files exist (copied from `new_data` templates), then batch NSGA-II for all nested scenarios:

```bash
python3 scripts/ensure_tariff_sensitivity_overall.py
python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --list-cases
python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --cases all
# smoke test (tiny budget, one case):
python3 run_tariff_sensitivity_batch.py --drought pers87_sev0.83n_4 --quick --max-cases 1
```

**Fixed desalination utilization** (no RBF policy optimization in that script’s inner loop — see module docstring):

```bash
python3 fixed_desal_experiment/run_all_cases_fixed.py --drought pers87_sev0.83n_4 --cases all --fractions 1.0,0.8,0.6,0.4,0.2
```

**Reinforcement learning** (on `RL-layer`, after RL dependencies are installed):

```bash
python3 src/train_rl.py
python3 src/visualize_rl.py
python3 src/compare_pareto_rl.py
```

---

## Acknowledgments

- **Original model and research codebase:** Prof. **Marta Zaniolo**, ZEDD Lab, Duke University.
- **This fork:** workflow, cost-curve tooling, performance-oriented refactors, batch runners, fixed-desal experiments, and the optional RL layer as described above.

For questions about the **upstream scientific intent** of the Santa Barbara formulation, refer to Prof. Zaniolo’s publications and lab materials. For **fork-specific** usage issues, open an issue on this repository.
