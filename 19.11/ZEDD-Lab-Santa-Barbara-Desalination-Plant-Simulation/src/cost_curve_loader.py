#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost Curve Loader for Custom CSV-based Cost Curves

This module loads and manages cost curve data from CSV files instead of using
linear mx+b cost curves.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional, Union

class CostCurveLoader:
    """Loads and manages cost curve data from CSV files."""
    
    def __init__(self, cost_curves_dir: str = "cost_curves"):
        """
        Initialize the cost curve loader.
        
        Args:
            cost_curves_dir: Directory containing the cost curve CSV files
        """
        # Allow selecting a specific cost-curve subset (e.g., "new_data" or "old_data")
        # via the COST_CURVES_SET environment variable, without changing call sites.
        subset = os.environ.get("COST_CURVES_SET")
        if subset:
            cost_curves_dir = os.path.join(cost_curves_dir, subset)

        self.cost_curves_dir = cost_curves_dir
        self.metadata = self._load_metadata()
        self._metadata_case_map = self._build_metadata_case_map()
        self._case_files = self._discover_case_files()
        self._cache = {}  # Cache for loaded cost curves
        
    def _load_metadata(self) -> Optional[pd.DataFrame]:
        """Load the metadata CSV file if available."""
        metadata_path = os.path.join(self.cost_curves_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        return None

    def _build_metadata_case_map(self) -> Dict[str, Dict]:
        """Create a lookup map from case identifier (as string) to metadata row."""
        if self.metadata is None:
            return {}

        case_map = {}
        for _, row in self.metadata.iterrows():
            case_id = str(row['case_number']).strip()
            case_map[case_id] = row.to_dict()
        return case_map

    def _discover_case_files(self) -> Dict[str, Dict[str, str]]:
        """Discover available case files by scanning the tree under ``cost_curves_dir``.

        A case is any directory that contains ``{stem}_summer.csv``, ``{stem}_winter.csv``,
        and ``{stem}_overall.csv`` for the same ``stem``. The case id is
        ``relpath/stem`` using ``/`` separators (e.g. ``new_data/basetariff_baseline/3mpd_30vessels``).
        """
        case_files: Dict[str, Dict[str, str]] = {}
        if not os.path.isdir(self.cost_curves_dir):
            return case_files

        skip_dir_names = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}

        for dirpath, dirnames, filenames in os.walk(self.cost_curves_dir):
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".") and d not in skip_dir_names
            ]
            for filename in filenames:
                if not filename.endswith("_summer.csv"):
                    continue
                stem = filename[: -len("_summer.csv")]
                overall_name = f"{stem}_overall.csv"
                winter_name = f"{stem}_winter.csv"
                if overall_name not in filenames or winter_name not in filenames:
                    continue
                summer_path = os.path.join(dirpath, filename)
                winter_path = os.path.join(dirpath, winter_name)
                overall_path = os.path.join(dirpath, overall_name)
                rel = os.path.relpath(dirpath, self.cost_curves_dir)
                if rel in (".", ""):
                    case_id = stem
                else:
                    case_id = os.path.join(rel, stem).replace("\\", "/")
                case_files[case_id] = {
                    "overall": overall_path,
                    "summer": summer_path,
                    "winter": winter_path,
                }

        return case_files

    def _normalize_case_id(self, case_id: Union[int, str]) -> str:
        """Normalize case identifiers to string form used internally."""
        if isinstance(case_id, (int, float)) and not isinstance(case_id, bool):
            s = str(int(case_id))
        else:
            s = str(case_id).strip()
        if s in self._case_files:
            return s
        # When scanning the full ``cost_curves/`` tree, cases live under ``new_data/`` or
        # ``old_data/``; allow the historical ids ``basetariff_*/stem`` without the prefix.
        for prefix in ("new_data", "old_data"):
            cand = f"{prefix}/{s}"
            if cand in self._case_files:
                return cand
        return s
    
    def get_available_cases(self) -> list:
        """Get list of available case identifiers (ints when possible)."""
        cases = sorted(self._case_files.keys(), key=lambda x: (not x.isdigit(), x if not x.isdigit() else int(x)))

        normalized = []
        for case in cases:
            if case.isdigit():
                normalized.append(int(case))
            else:
                normalized.append(case)
        return normalized
    
    def get_case_info(self, case_number: Union[int, str]) -> dict:
        """Get information about a specific case, falling back to filenames when metadata is absent."""
        case_id = self._normalize_case_id(case_number)

        if case_id not in self._case_files:
            raise ValueError(f"Case '{case_number}' not found in directory {self.cost_curves_dir}")

        metadata = self._metadata_case_map.get(case_id)
        if metadata is not None:
            return metadata

        # Minimal info when metadata is unavailable
        return {
            'case_number': case_id,
            'ro_capacity': None,
            'scenario': None,
            'summer_winter_split': None,
        }
    
    def load_cost_curve(self, case_number: Union[int, str]) -> dict:
        """
        Load cost curve data for a specific case.
        
        Args:
            case_number: The case number to load (1-45)
            
        Returns:
            Dictionary containing cost curve data with keys:
            - 'overall': Overall cost data (capital, labor)
            - 'summer': Summer cost curve data
            - 'winter': Winter cost curve data
        """
        case_id = self._normalize_case_id(case_number)

        if case_id not in self._case_files:
            raise ValueError(f"Case '{case_number}' not found in directory {self.cost_curves_dir}")

        if case_id in self._cache:
            return self._cache[case_id]
        
        paths = self._case_files[case_id]

        overall_df = pd.read_csv(paths["overall"])
        summer_df = pd.read_csv(paths["summer"])
        winter_df = pd.read_csv(paths["winter"])

        def _col(df: pd.DataFrame, *candidates: str) -> pd.Series:
            lower = {c.lower(): c for c in df.columns}
            for name in candidates:
                key = name.lower()
                if key in lower:
                    return df[lower[key]]
            for name in candidates:
                for col in df.columns:
                    if name.lower() in col.lower().replace(" ", ""):
                        return df[col]
            raise KeyError(
                f"None of {candidates} found in columns {list(df.columns)} "
                f"(file context: {paths})"
            )

        cap_col = _col(overall_df, "capital_upgrade_cost_usd")
        lab_col = _col(overall_df, "labor_cost_usd_month")

        cost_curve_data = {
            "overall": {
                "capital_upgrade_cost_usd": float(cap_col.iloc[0]),
                "labor_cost_usd_month": float(lab_col.iloc[0]),
            },
            "summer": {
                "water_production": _col(
                    summer_df, "water_production_AF_month"
                ).values.astype(float),
                "electricity_cost": _col(
                    summer_df, "electricity_cost_usd_month"
                ).values.astype(float),
                "fixed_cost": _col(summer_df, "fixed_cost_usd_month").values.astype(
                    float
                ),
            },
            "winter": {
                "water_production": _col(
                    winter_df, "water_production_AF_month"
                ).values.astype(float),
                "electricity_cost": _col(
                    winter_df, "electricity_cost_usd_month"
                ).values.astype(float),
                "fixed_cost": _col(winter_df, "fixed_cost_usd_month").values.astype(
                    float
                ),
            },
        }
        
        # Cache the data
        self._cache[case_id] = cost_curve_data
        return cost_curve_data
    
    def get_cost_for_production(self, case_number: Union[int, str], production: float, 
                              is_summer: bool = True) -> Tuple[float, float]:
        """
        Get the cost for a given water production level.
        
        Args:
            case_number: The case number
            production: Water production in AF/month
            is_summer: Whether to use summer (True) or winter (False) cost curve
            
        Returns:
            Tuple of (electricity_cost, fixed_cost) in USD/month
        """
        cost_data = self.load_cost_curve(case_number)
        season = 'summer' if is_summer else 'winter'
        
        water_prod = cost_data[season]['water_production']
        elec_cost = cost_data[season]['electricity_cost']
        fixed_cost = cost_data[season]['fixed_cost']
        
        # Interpolate to get cost for the given production level
        if production <= water_prod[0]:
            return float(elec_cost[0]), float(fixed_cost[0])
        elif production >= water_prod[-1]:
            return float(elec_cost[-1]), float(fixed_cost[-1])
        else:
            # Linear interpolation
            elec_cost_interp = np.interp(production, water_prod, elec_cost)
            fixed_cost_interp = np.interp(production, water_prod, fixed_cost)
            return float(elec_cost_interp), float(fixed_cost_interp)
    
    def get_capital_cost(self, case_number: Union[int, str]) -> float:
        """Get the capital upgrade cost for a case."""
        cost_data = self.load_cost_curve(case_number)
        return cost_data['overall']['capital_upgrade_cost_usd']
 
    def get_capital_cost_amortized(
        self,
        case_number: Union[int, str],
        amortization_years: float = 30.0,
        period: str = "annual",
    ) -> float:
        """Return the amortized capital cost for the requested period.

        Args:
            case_number: Case identifier.
            amortization_years: Payback period (must be > 0).

        Returns:
            Amortized cost in USD per selected period.
        """
        if amortization_years <= 0:
            raise ValueError("amortization_years must be positive")

        annual_cost = self.get_capital_cost(case_number) / amortization_years

        if period == "annual":
            return annual_cost
        if period == "monthly":
            return annual_cost / 12.0

        raise ValueError("period must be either 'annual' or 'monthly'")

    def get_labor_cost(self, case_number: Union[int, str]) -> float:
        """Get the monthly labor cost for a case."""
        cost_data = self.load_cost_curve(case_number)
        return cost_data['overall']['labor_cost_usd_month']
    
    def get_max_production(self, case_number: Union[int, str], is_summer: bool = True) -> float:
        """Get the maximum water production capacity for a case."""
        cost_data = self.load_cost_curve(case_number)
        season = 'summer' if is_summer else 'winter'
        return float(cost_data[season]['water_production'][-1])
