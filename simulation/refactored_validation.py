#!/usr/bin/env python3
"""
Input validation and business logic checks for simulation parameters.
Ensures parameter combinations make business sense and prevents simulation failures.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional


def preflight_validate(cfg: dict) -> bool:
    """
    Validate configuration before simulation runs.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        bool: True if config is valid
    """
    errs = []
    
    if "USAGE_SHARE" in cfg and not isinstance(cfg["USAGE_SHARE"], (dict, list, tuple)):
        errs.append("USAGE_SHARE must be a dict/list")
    
    if "STATIONS" in cfg and not isinstance(cfg["STATIONS"], (dict, list, int)):
        errs.append("STATIONS must be dict/list/int")
    
    # Business logic validations
    rent = cfg.get("MONTHLY_RENT", 0)
    membership_price = cfg.get("MEMBERSHIP_PRICE", 0)
    capacity = cfg.get("STUDIO_CAPACITY", 0)
    
    if rent > membership_price * capacity * 0.7:
        errs.append(f"Rent (${rent}) may be too high relative to potential revenue")
    
    if errs:
        st.error("Invalid inputs:\n- " + "\n- ".join(errs))
        return False
    
    return True


def normalize_capex_items(df) -> List[Dict[str, Any]]:
    """
    Convert the data_editor DataFrame into a clean list of capex items.
    
    Args:
        df: DataFrame from st.data_editor
    
    Returns:
        List of normalized capex item dictionaries
    """
    items = []
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return items
    
    for _, r in df.iterrows():
        try:
            label = str(r.get("label", "")).strip()
            unit = float(r.get("unit_cost", 0) or 0)
            cnt = int(r.get("count", 1) or 1)
            mth = r.get("month", None)
            thr = r.get("member_threshold", None)
            enabled = bool(r.get("enabled", True))
            
            # Clean up None/empty values
            mth = None if (mth == "" or pd.isna(mth)) else int(mth)
            thr = None if (thr == "" or pd.isna(thr)) else int(thr)
            
            if not enabled:
                continue
                
            if unit > 0 and (mth is not None or thr is not None):
                items.append({
                    "label": label,
                    "unit_cost": unit,
                    "count": cnt,
                    "month": mth,
                    "member_threshold": thr,
                    "finance_504": bool(r.get("finance_504", False)),
                })
        except Exception:
            continue
    
    return items


def normalize_market_inflow(d: dict) -> dict:
    """
    Normalize market inflow data to ensure valid values.
    
    Args:
        d: Market inflow dictionary
    
    Returns:
        Normalized market inflow dictionary
    """
    pools = {
        "community_studio": d.get("community_studio", 0),
        "home_studio": d.get("home_studio", 0),
        "no_access": d.get("no_access", 0),
    }
    
    out = {}
    for k, v in pools.items():
        try:
            out[k] = max(0, int(v))
        except Exception:
            out[k] = 0
    
    return out


def validate_loan_configuration(loan_config: dict) -> List[str]:
    """
    Validate loan configuration for business viability.
    
    Args:
        loan_config: Dictionary with loan parameters
    
    Returns:
        List of validation warnings
    """
    warnings = []
    
    # Check debt service coverage ratios
    monthly_debt_service = loan_config.get('monthly_payment', 0)
    projected_revenue = loan_config.get('projected_monthly_revenue', 0)
    
    if monthly_debt_service > projected_revenue * 0.3:
        warnings.append("Debt service exceeds 30% of projected revenue - may be unsustainable")
    
    # Check loan-to-value ratios
    total_loan = loan_config.get('total_principal', 0)
    asset_value = loan_config.get('asset_value', 0)
    
    if total_loan > asset_value * 0.9:
        warnings.append("Loan-to-value ratio exceeds 90% - may not qualify for SBA financing")
    
    return warnings


def check_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    """
    Check if DataFrame has required columns.
    
    Args:
        df: DataFrame to check
        required: List of required column names
    
    Returns:
        List of missing column names
    """
    return [col for col in required if col not in df.columns]


def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Pick the first available column from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of candidate column names
    
    Returns:
        First matching column name, or None if none found
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None
