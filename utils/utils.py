#!/usr/bin/env python3
"""
Utility functions and helpers used across the application.
Contains common functions for data processing and calculations.
"""

import streamlit as st
from modular_simulator import get_default_cfg
from typing import Optional


@st.cache_data(show_spinner=False)
def get_defaults_cached():
    """
    Cached wrapper for getting default configuration from modular_simulator.
    
    Returns:
        Default configuration dictionary
    """
    return get_default_cfg()


def format_currency(amount: float, thousands_sep: bool = True) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Amount to format
        thousands_sep: Whether to include thousands separator
    
    Returns:
        Formatted currency string
    """
    if thousands_sep:
        return f"${amount:,.0f}"
    else:
        return f"${amount:.0f}"


def format_percentage(value: float, decimal_places: int = 1) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Value to format (0.15 = 15%)
        decimal_places: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimal_places}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
    
    Returns:
        Division result or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
    
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Calculate monthly loan payment using standard amortization formula.
    
    Args:
        principal: Loan principal amount
        annual_rate: Annual interest rate (as decimal, e.g., 0.08 for 8%)
        years: Loan term in years
    
    Returns:
        Monthly payment amount
    """
    if annual_rate == 0:
        return principal / (years * 12)
    
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    return principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)


def interpolate_linear(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Linear interpolation between two points.
    
    Args:
        x: X value to interpolate
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
    
    Returns:
        Interpolated Y value
    """
    if x2 == x1:
        return y1
    
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def calculate_roi(initial_investment: float, final_value: float, time_period: float) -> float:
    """
    Calculate annualized return on investment.
    
    Args:
        initial_investment: Initial investment amount
        final_value: Final value of investment
        time_period: Time period in years
    
    Returns:
        Annualized ROI as decimal
    """
    if initial_investment <= 0 or time_period <= 0:
        return 0.0
    
    total_return = final_value / initial_investment
    return (total_return ** (1 / time_period)) - 1


def get_business_day_count(start_month: int, end_month: int, business_days_per_month: int = 22) -> int:
    """
    Calculate business days between two months.
    
    Args:
        start_month: Starting month (0-based)
        end_month: Ending month (0-based)
        business_days_per_month: Average business days per month
    
    Returns:
        Total business days
    """
    if end_month <= start_month:
        return 0
    
    return (end_month - start_month) * business_days_per_month
