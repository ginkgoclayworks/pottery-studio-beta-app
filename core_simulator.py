#!/usr/bin/env python3
"""
core_simulator.py

Wrapper around the existing simulation engine that provides a simplified interface
for the beta app while preserving all the sophisticated modeling capabilities.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, NamedTuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import the existing adapter
try:
    from final_batch_adapter import run_original_once
except ImportError:
    # Add the parent directory to path if needed
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    from final_batch_adapter import run_original_once

from config_manager import StudioConfig


@dataclass
class SimulationSummary:
    """Simplified results focused on business-relevant metrics"""
    
    # Financial health
    survival_probability: float
    median_final_cash: float
    cash_runway_months: float
    
    # Business performance
    median_monthly_revenue: float
    median_monthly_profit: float
    break_even_month: float
    
    # Risk metrics
    cash_10th_percentile: float
    worst_case_runway: float
    loan_stress_probability: float
    
    # Growth metrics
    final_member_count: float
    revenue_growth_rate: float
    
    # Loan servicing
    debt_service_coverage_ratio: float
    monthly_loan_payment: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            "survival_probability": self.survival_probability,
            "median_final_cash": self.median_final_cash,
            "cash_runway_months": self.cash_runway_months,
            "median_monthly_revenue": self.median_monthly_revenue,
            "median_monthly_profit": self.median_monthly_profit,
            "break_even_month": self.break_even_month,
            "cash_10th_percentile": self.cash_10th_percentile,
            "worst_case_runway": self.worst_case_runway,
            "loan_stress_probability": self.loan_stress_probability,
            "final_member_count": self.final_member_count,
            "revenue_growth_rate": self.revenue_growth_rate,
            "debt_service_coverage_ratio": self.debt_service_coverage_ratio,
            "monthly_loan_payment": self.monthly_loan_payment,
        }


class SimulationResults:
    """
    Container for simulation results that provides both simplified summary
    metrics for beta users and access to detailed data for advanced analysis.
    """
    
    def __init__(self, raw_dataframe: pd.DataFrame, config: StudioConfig):
        self.raw_data = raw_dataframe
        self.config = config
        self._summary = None
        self._validate_data()
    
    def _validate_data(self):
        """Ensure the raw data has expected columns"""
        required_cols = ['month', 'simulation_id', 'cash_balance']
        missing = [col for col in required_cols if col not in self.raw_data.columns]
        if missing:
            raise ValueError(f"Simulation data missing required columns: {missing}")
    
    @property
    def summary(self) -> SimulationSummary:
        """Get business-focused summary metrics"""
        if self._summary is None:
            self._summary = self._compute_summary()
        return self._summary
    
    def _compute_summary(self) -> SimulationSummary:
        """Compute summary metrics from raw simulation data"""
        df = self.raw_data
        final_month = df['month'].max()
        
        # Group by simulation for aggregate metrics
        sim_groups = df.groupby('simulation_id')
        
        # Financial health metrics
        final_cash = df[df['month'] == final_month]['cash_balance']
        survival_prob = float((final_cash > 0).mean())
        median_final_cash = float(final_cash.median())
        cash_10th_pct = float(final_cash.quantile(0.10))
        
        # Cash runway calculation (months until insolvency)
        def months_until_insolvency(group):
            negative_months = group[group['cash_balance'] < 0]['month']
            return negative_months.min() if len(negative_months) > 0 else final_month + 1
        
        runway_by_sim = sim_groups.apply(months_until_insolvency)
        median_runway = float(runway_by_sim.median())
        worst_case_runway = float(runway_by_sim.quantile(0.10))
        
        # Revenue and profitability
        revenue_col = self._find_column(['revenue_total', 'total_revenue', 'revenue'])
        profit_col = self._find_column(['net_profit', 'operating_profit', 'profit'])
        
        if revenue_col:
            median_monthly_revenue = float(df[revenue_col].median())
            # Calculate growth rate from first to last year
            first_year_revenue = df[df['month'] <= 12][revenue_col].sum()
            last_year_start = max(1, final_month - 11)
            last_year_revenue = df[df['month'] >= last_year_start][revenue_col].sum()
            revenue_growth_rate = float((last_year_revenue / max(first_year_revenue, 1) - 1) if first_year_revenue > 0 else 0)
        else:
            median_monthly_revenue = 0.0
            revenue_growth_rate = 0.0
        
        if profit_col:
            median_monthly_profit = float(df[profit_col].median())
        else:
            median_monthly_profit = 0.0
        
        # Break-even analysis
        if profit_col:
            def find_breakeven(group):
                profitable_months = group[group[profit_col] > 0]['month']
                return profitable_months.min() if len(profitable_months) > 0 else np.nan
            
            breakeven_by_sim = sim_groups.apply(find_breakeven)
            break_even_month = float(breakeven_by_sim.median()) if not breakeven_by_sim.isna().all() else np.nan
        else:
            break_even_month = np.nan
        
        # Member count
        member_col = self._find_column(['members', 'member_count', 'total_members'])
        if member_col:
            final_member_count = float(df[df['month'] == final_month][member_col].median())
        else:
            final_member_count = 0.0
        
        # Loan servicing metrics
        dscr_col = self._find_column(['dscr', 'debt_service_coverage_ratio'])
        payment_col = self._find_column(['loan_payment_total', 'total_loan_payment', 'loan_payment'])
        
        if dscr_col:
            median_dscr = float(df[dscr_col].median())
            loan_stress_prob = float((df[dscr_col] < 1.25).mean())  # Below 1.25 is concerning for lenders
        else:
            median_dscr = np.nan
            loan_stress_prob = np.nan
        
        if payment_col:
            monthly_loan_payment = float(df[payment_col].median())
        else:
            monthly_loan_payment = 0.0
        
        return SimulationSummary(
            survival_probability=survival_prob,
            median_final_cash=median_final_cash,
            cash_runway_months=median_runway,
            median_monthly_revenue=median_monthly_revenue,
            median_monthly_profit=median_monthly_profit,
            break_even_month=break_even_month,
            cash_10th_percentile=cash_10th_pct,
            worst_case_runway=worst_case_runway,
            loan_stress_probability=loan_stress_prob,
            final_member_count=final_member_count,
            revenue_growth_rate=revenue_growth_rate,
            debt_service_coverage_ratio=median_dscr,
            monthly_loan_payment=monthly_loan_payment,
        )
    
    def _find_column(self, candidates: list) -> Optional[str]:
        """Find the first available column from a list of candidates"""
        for col in candidates:
            if col in self.raw_data.columns:
                return col
        return None
    
    def get_monthly_timeseries(self, metric: str) -> pd.DataFrame:
        """
        Get monthly timeseries data for a specific metric with percentiles
        
        Args:
            metric: Column name to extract (e.g., 'cash_balance', 'revenue_total')
            
        Returns:
            DataFrame with columns: month, p10, p25, p50, p75, p90, mean
        """
        if metric not in self.raw_data.columns:
            raise ValueError(f"Metric '{metric}' not found in simulation data")
        
        monthly_stats = (self.raw_data.groupby('month')[metric]
                        .agg(['mean', 'std', 'count'])
                        .reset_index())
        
        monthly_percentiles = (self.raw_data.groupby('month')[metric]
                              .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                              .unstack()
                              .reset_index())
        
        monthly_percentiles.columns = ['month', 'p10', 'p25', 'p50', 'p75', 'p90']
        
        result = monthly_stats.merge(monthly_percentiles, on='month')
        return result
    
    def get_risk_analysis(self) -> Dict[str, Any]:
        """Get detailed risk analysis including stress testing scenarios"""
        df = self.raw_data
        
        # Cash flow stress analysis
        min_cash_by_sim = df.groupby('simulation_id')['cash_balance'].min()
        
        risk_metrics = {
            "cash_never_negative_prob": float((min_cash_by_sim >= 0).mean()),
            "cash_below_10k_prob": float((min_cash_by_sim < 10000).mean()),
            "cash_below_25k_prob": float((min_cash_by_sim < 25000).mean()),
            "minimum_cash_p10": float(min_cash_by_sim.quantile(0.10)),
            "minimum_cash_p50": float(min_cash_by_sim.quantile(0.50)),
        }
        
        # DSCR analysis if available
        dscr_col = self._find_column(['dscr', 'debt_service_coverage_ratio'])
        if dscr_col:
            min_dscr_by_sim = df.groupby('simulation_id')[dscr_col].min()
            risk_metrics.update({
                "dscr_always_above_125_prob": float((min_dscr_by_sim >= 1.25).mean()),
                "dscr_always_above_100_prob": float((min_dscr_by_sim >= 1.00).mean()),
                "minimum_dscr_p10": float(min_dscr_by_sim.quantile(0.10)),
                "minimum_dscr_p50": float(min_dscr_by_sim.quantile(0.50)),
            })
        
        return risk_metrics


def run_simulation(config: StudioConfig, script_path: str = "modular_simulator.py") -> SimulationResults:
    """
    Run the pottery studio simulation with the provided configuration.
    
    Args:
        config: StudioConfig object with all simulation parameters
        script_path: Path to the main simulator script
        
    Returns:
        SimulationResults object with summary metrics and detailed data
        
    Raises:
        ValueError: If configuration validation fails
        RuntimeError: If simulation execution fails
    """
    
    # Validate configuration
    errors = config.validate_all()
    if errors:
        error_msg = "Configuration validation failed:\n"
        for section, section_errors in errors.items():
            error_msg += f"  {section}: {', '.join(section_errors)}\n"
        raise ValueError(error_msg)
    
    # Convert to legacy format
    legacy_params = config.to_legacy_format()
    
    try:
        # Run the existing simulator
        result = run_original_once(script_path, legacy_params)
        
        # Handle different return formats from the simulator
        if isinstance(result, tuple):
            raw_df, effective_config = result
        else:
            raw_df = result
            effective_config = None
        
        if raw_df is None or raw_df.empty:
            raise RuntimeError("Simulation returned empty results")
        
        # Wrap in our results class
        return SimulationResults(raw_df, config)
        
    except Exception as e:
        raise RuntimeError(f"Simulation execution failed: {str(e)}")


def run_scenario_comparison(scenarios: Dict[str, StudioConfig], 
                          script_path: str = "modular_simulator.py") -> Dict[str, SimulationResults]:
    """
    Run multiple scenarios and return results for comparison.
    
    Args:
        scenarios: Dictionary mapping scenario names to StudioConfig objects
        script_path: Path to the main simulator script
        
    Returns:
        Dictionary mapping scenario names to SimulationResults
    """
    results = {}
    
    for scenario_name, config in scenarios.items():
        try:
            results[scenario_name] = run_simulation(config, script_path)
        except Exception as e:
            print(f"Warning: Scenario '{scenario_name}' failed: {e}")
            continue
    
    return results


def create_quick_comparison_table(results: Dict[str, SimulationResults]) -> pd.DataFrame:
    """
    Create a simple comparison table of key metrics across scenarios.
    
    Args:
        results: Dictionary of scenario results
        
    Returns:
        DataFrame with scenarios as rows and key metrics as columns
    """
    comparison_data = []
    
    for scenario_name, result in results.items():
        summary = result.summary
        comparison_data.append({
            "Scenario": scenario_name,
            "Survival Rate": f"{summary.survival_probability:.1%}",
            "Final Cash": f"${summary.median_final_cash:,.0f}",
            "Runway (months)": f"{summary.cash_runway_months:.1f}",
            "Monthly Revenue": f"${summary.median_monthly_revenue:,.0f}",
            "Monthly Profit": f"${summary.median_monthly_profit:,.0f}",
            "Break-even (month)": f"{summary.break_even_month:.1f}" if not np.isnan(summary.break_even_month) else "N/A",
            "Final Members": f"{summary.final_member_count:.0f}",
            "DSCR": f"{summary.debt_service_coverage_ratio:.2f}" if not np.isnan(summary.debt_service_coverage_ratio) else "N/A",
        })
    
    return pd.DataFrame(comparison_data)


# Convenience functions for common analysis patterns
def run_sensitivity_analysis(base_config: StudioConfig, 
                           parameter_variations: Dict[str, Dict[str, Any]],
                           script_path: str = "modular_simulator.py") -> Dict[str, SimulationResults]:
    """
    Run sensitivity analysis by varying specific parameters around a base configuration.
    
    Args:
        base_config: Base configuration to modify
        parameter_variations: Dict of variation_name -> {section: {param: value}}
        script_path: Path to simulator script
        
    Returns:
        Dictionary of variation results
    """
    scenarios = {"baseline": base_config}
    
    for variation_name, modifications in parameter_variations.items():
        # Create a copy of the base config
        import copy
        varied_config = copy.deepcopy(base_config)
        
        # Apply modifications
        for section_name, param_changes in modifications.items():
            section = getattr(varied_config, section_name)
            for param_name, new_value in param_changes.items():
                setattr(section, param_name, new_value)
        
        scenarios[variation_name] = varied_config
    
    return run_scenario_comparison(scenarios, script_path)


# Example usage and testing functions
def run_example_simulation() -> SimulationResults:
    """Run an example simulation with default parameters for testing"""
    config = StudioConfig()
    
    # Reduce simulation time for quick testing
    config.simulation.num_simulations = 50
    config.simulation.time_horizon_months = 24
    
    return run_simulation(config)


if __name__ == "__main__":
    # Quick test of the wrapper
    print("Testing simulation wrapper...")
    
    try:
        result = run_example_simulation()
        print("✓ Simulation completed successfully")
        
        summary = result.summary
        print(f"✓ Survival probability: {summary.survival_probability:.1%}")
        print(f"✓ Median final cash: ${summary.median_final_cash:,.0f}")
        print(f"✓ Cash runway: {summary.cash_runway_months:.1f} months")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)