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
from typing import Dict, Tuple, Optional

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
        self._cache = {}  # Cache for loaded cost curves
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load the metadata CSV file to understand case parameters."""
        metadata_path = os.path.join(self.cost_curves_dir, "metadata.csv")
        return pd.read_csv(metadata_path)
    
    def get_available_cases(self) -> list:
        """Get list of available case numbers."""
        return self.metadata['case_number'].tolist()
    
    def get_case_info(self, case_number: int) -> dict:
        """Get information about a specific case."""
        case_info = self.metadata[self.metadata['case_number'] == case_number]
        if case_info.empty:
            raise ValueError(f"Case {case_number} not found in metadata")
        
        return {
            'case_number': int(case_info['case_number'].iloc[0]),
            'ro_capacity': float(case_info['ro_capacity'].iloc[0]),
            'scenario': case_info['scenario'].iloc[0],
            'summer_winter_split': float(case_info['summer_winter_split'].iloc[0])
        }
    
    def load_cost_curve(self, case_number: int) -> dict:
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
        if case_number in self._cache:
            return self._cache[case_number]
        
        # Load overall costs (capital and labor)
        overall_path = os.path.join(self.cost_curves_dir, f"{case_number}_overall.csv")
        overall_df = pd.read_csv(overall_path)
        
        # Load summer cost curve
        summer_path = os.path.join(self.cost_curves_dir, f"{case_number}_summer.csv")
        summer_df = pd.read_csv(summer_path)
        
        # Load winter cost curve
        winter_path = os.path.join(self.cost_curves_dir, f"{case_number}_winter.csv")
        winter_df = pd.read_csv(winter_path)
        
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
        self._cache[case_number] = cost_curve_data
        return cost_curve_data
    
    def get_cost_for_production(self, case_number: int, production: float, 
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
    
    def get_capital_cost(self, case_number: int) -> float:
        """Get the capital upgrade cost for a case."""
        cost_data = self.load_cost_curve(case_number)
        return cost_data['overall']['capital_upgrade_cost_usd']
 
    def get_capital_cost_amortized(
        self,
        case_number: int,
        amortization_years: float = 30.0,
        period: str = "annual",
    ) -> float:
        """Return the amortized capital cost for the requested period.

        Args:
            case_number: Case identifier.
            amortization_years: Payback period (must be > 0).
            period: Either "annual" or "monthly".

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

    def get_labor_cost(self, case_number: int) -> float:
        """Get the monthly labor cost for a case."""
        cost_data = self.load_cost_curve(case_number)
        return cost_data['overall']['labor_cost_usd_month']
    
    def get_max_production(self, case_number: int, is_summer: bool = True) -> float:
        """Get the maximum water production capacity for a case."""
        cost_data = self.load_cost_curve(case_number)
        season = 'summer' if is_summer else 'winter'
        return float(cost_data[season]['water_production'][-1])
