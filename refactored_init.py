# config/__init__.py
"""Configuration module for pottery studio simulator."""

from .parameters import CONSOLIDATED_PARAM_SPECS, CONSOLIDATED_GROUPS, consolidate_build_overrides
from .scenarios import SCENARIOS, STRATEGIES, DEFAULT_CAPEX_ITEMS

__all__ = [
    'CONSOLIDATED_PARAM_SPECS',
    'CONSOLIDATED_GROUPS', 
    'consolidate_build_overrides',
    'SCENARIOS',
    'STRATEGIES',
    'DEFAULT_CAPEX_ITEMS'
]

# simulation/__init__.py
"""Simulation engine module."""

from .engine import run_cell_cached
from .validation import preflight_validate, normalize_capex_items, normalize_market_inflow

__all__ = [
    'run_cell_cached',
    'preflight_validate',
    'normalize_capex_items',
    'normalize_market_inflow'
]

# ui/__init__.py
"""User interface components module."""

from .components import (
    render_parameter_group,
    render_parameter,
    render_loan_controls,
    render_simulation_settings
)

__all__ = [
    'render_parameter_group',
    'render_parameter', 
    'render_loan_controls',
    'render_simulation_settings'
]

# analysis/__init__.py
"""Analysis and metrics module."""

from .metrics import compute_kpis_from_cell, summarize_cell, calculate_business_viability_score

__all__ = [
    'compute_kpis_from_cell',
    'summarize_cell',
    'calculate_business_viability_score'
]

# visualization/__init__.py
"""Visualization and chart generation module."""

from .charts import FigureCapture, create_safe_heatmap, apply_visualization_patches

__all__ = [
    'FigureCapture',
    'create_safe_heatmap', 
    'apply_visualization_patches'
]

# utils/__init__.py
"""Utilities and helper functions module."""

from .helpers import (
    get_defaults_cached,
    format_currency,
    format_percentage,
    safe_divide,
    calculate_loan_payment
)

__all__ = [
    'get_defaults_cached',
    'format_currency',
    'format_percentage', 
    'safe_divide',
    'calculate_loan_payment'
]

# main.py - New main entry point
#!/usr/bin/env python3
"""
Main entry point for the pottery studio simulator.
Run with: streamlit run main.py
"""

from ui.main import main

if __name__ == "__main__":
    main()
