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
import re
from typing import Dict, Tuple, Optional, Union

class CostCurveLoader:
    """Loads and manages cost curve data from CSV files."""
    
    def __init__(self, cost_curves_dir: str = "cost_curves"):
        """
        Initialize the cost curve loader.
        
        Args:
            cost_curves_dir: Directory containing the cost curve CSV files
        """
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
        """Discover available case files by scanning the directory."""
        case_files: Dict[str, Dict[str, str]] = {}
        if not os.path.isdir(self.cost_curves_dir):
            return case_files

        for filename in os.listdir(self.cost_curves_dir):
            if not filename.endswith('_summer.csv'):
                continue

            case_id = filename[:-len('_summer.csv')]
            overall_name = f"{case_id}_overall.csv"
            winter_name = f"{case_id}_winter.csv"

            overall_path = os.path.join(self.cost_curves_dir, overall_name)
            summer_path = os.path.join(self.cost_curves_dir, filename)
            winter_path = os.path.join(self.cost_curves_dir, winter_name)

            if os.path.exists(overall_path) and os.path.exists(winter_path):
                case_files[case_id] = {
                    'overall': overall_path,
                    'summer': summer_path,
                    'winter': winter_path,
                }

        return case_files

    def _normalize_case_id(self, case_id: Union[int, str]) -> str:
        """Normalize case identifiers to string form used internally."""
        if isinstance(case_id, (int, float)) and not isinstance(case_id, bool):
            return str(int(case_id))
        return str(case_id).strip()
    
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

        overall_df = pd.read_csv(paths['overall'])
        summer_df = pd.read_csv(paths['summer'])
        winter_df = pd.read_csv(paths['winter'])
        
        cost_curve_data = {
            'overall': {
                'capital_upgrade_cost_usd': float(overall_df['capital_upgrade_cost_usd'].iloc[0]),
                'labor_cost_usd_month': float(overall_df['labor_cost_usd_month'].iloc[0])
            },
            'summer': {
                'water_production': summer_df['water_production_AF_month'].values,
                'electricity_cost': summer_df['electricity_cost_usd_month'].values,
                'fixed_cost': summer_df['fixed_cost_usd_month'].values
            },
            'winter': {
                'water_production': winter_df['water_production_AF_month'].values,
                'electricity_cost': winter_df['electricity_cost_usd_month'].values,
                'fixed_cost': winter_df['fixed_cost_usd_month'].values
            }
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
    
    def get_cost_bounds(self, case_number: Union[int, str], 
                       amortization_years: float = 30.0) -> Tuple[float, float]:
        """
        Get theoretical min and max monthly costs for a case based on CSV data.
        
        Args:
            case_number: Case identifier
            amortization_years: Years over which to amortize capital cost
            
        Returns:
            Tuple of (min_cost, max_cost) in USD/month
        """
        cost_data = self.load_cost_curve(case_number)
        
        # Get capital cost amortized monthly
        capital_monthly = self.get_capital_cost_amortized(case_number, amortization_years, period="monthly")
        
        # Get labor cost (monthly)
        labor_monthly = self.get_labor_cost(case_number)
        
        # Get fixed and electricity costs from both seasons
        summer_fixed = cost_data['summer']['fixed_cost']
        summer_elec = cost_data['summer']['electricity_cost']
        winter_fixed = cost_data['winter']['fixed_cost']
        winter_elec = cost_data['winter']['electricity_cost']
        
        # Minimum cost: no production, use minimum fixed cost + capital (no labor if idle)
        # Take minimum across seasons
        min_fixed = min(np.min(summer_fixed), np.min(winter_fixed))
        min_cost = min_fixed + capital_monthly  # No production, no labor, no electricity
        
        # Maximum cost: maximum production with maximum costs + labor + capital
        # Take maximum across seasons
        max_fixed = max(np.max(summer_fixed), np.max(winter_fixed))
        max_elec = max(np.max(summer_elec), np.max(winter_elec))
        max_cost = max_fixed + max_elec + labor_monthly + capital_monthly
        
        return float(min_cost), float(max_cost)
    
    def parse_mpd_vessels(self, case_number: Union[int, str]) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract MPD and vessel count from case identifier if present.
        
        Args:
            case_number: Case identifier (e.g., '3mpd_30vessels' or '16')
            
        Returns:
            Tuple of (mpd, vessels) if parseable, else (None, None)
        """
        case_str = str(case_number).strip()
        
        # Try to match pattern like "3mpd_30vessels" or "4mpd_36vessels"
        match = re.match(r'^(\d+)mpd[_-](\d+)vessels?$', case_str, re.IGNORECASE)
        if match:
            mpd = float(match.group(1))
            vessels = float(match.group(2))
            return mpd, vessels
        
        # Check metadata if available
        case_id = self._normalize_case_id(case_number)
        if case_id in self._metadata_case_map:
            metadata = self._metadata_case_map[case_id]
            # If metadata has these fields, use them (assuming they exist)
            if 'mpd' in metadata and 'vessels' in metadata:
                try:
                    return float(metadata['mpd']), float(metadata['vessels'])
                except (ValueError, TypeError, KeyError):
                    pass
        
        return None, None
