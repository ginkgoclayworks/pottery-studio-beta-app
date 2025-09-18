#!/usr/bin/env python3
"""
KPI calculations and result processing for simulation analysis.
Computes business metrics, survival probabilities, and performance indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from simulation.validation import pick_column


def compute_kpis_from_cell(df_cell: pd.DataFrame) -> dict:
    """
    Compute lender-style KPIs from a single cell's simulation dataframe.
    
    Args:
        df_cell: DataFrame with simulation results
    
    Returns:
        Dictionary of computed KPIs
    """
    out = {}
    if df_cell.empty:
        return out

    # Resolve columns
    month_col = "month" if "month" in df_cell.columns else ("Month" if "Month" in df_cell.columns else "t")
    if month_col not in df_cell.columns:
        return out

    cash_col = pick_column(df_cell, ["cash_balance", "cash", "ending_cash"])
    if cash_col is None:
        return out

    last_month = int(df_cell[month_col].max())
    end = df_cell[df_cell[month_col] == last_month]

    sim_col = "simulation_id" if "simulation_id" in df_cell.columns else None
    if sim_col:
        min_cash_by_sim = df_cell.groupby(sim_col)[cash_col].min()
    else:
        min_cash_by_sim = pd.Series([float(df_cell[cash_col].min())])

    out["survival_prob"] = float((min_cash_by_sim >= 0).mean())
    out["cash_q10"] = float(end[cash_col].quantile(0.10))
    out["cash_med"] = float(end[cash_col].quantile(0.50))
    out["cash_q90"] = float(end[cash_col].quantile(0.90))

    if "dscr" in end.columns:
        out["dscr_q10"] = float(end["dscr"].quantile(0.10))
        out["dscr_med"] = float(end["dscr"].quantile(0.50))
        out["dscr_q90"] = float(end["dscr"].quantile(0.90))

    if "active_members" in end.columns:
        out["members_med"] = float(end["active_members"].median())

    return out


def summarize_cell(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """
    Summarize simulation results for a single cell.
    
    Args:
        df: DataFrame with simulation results
    
    Returns:
        Tuple of (summary dict, timings DataFrame)
    """
    if df.empty:
        return {}, pd.DataFrame()
    
    env_col = "environment"
    strat_col = "strategy"
    sim_col = "simulation_id"
    month_col = "month" if "month" in df.columns else ("Month" if "Month" in df.columns else "t")
    
    if month_col not in df.columns:
        return {}, pd.DataFrame()

    cash_col = pick_column(df, ["cash_balance","cash","ending_cash"])
    cf_col = pick_column(df, ["cfads","operating_cash_flow","op_cf","net_cash_flow","cash_flow"])

    if cash_col is None:
        raise RuntimeError("cash balance column not found in results.")
    if cf_col is None:
        df = df.sort_values([env_col, strat_col, sim_col, month_col]).copy()
        df["_fallback_cf"] = df.groupby([env_col, strat_col, sim_col])[cash_col].diff().fillna(0.0)
        cf_col = "_fallback_cf"
    
    breakeven_k = 3

    def _first_cash_negative(g: pd.DataFrame) -> float:
        s = g.set_index(month_col)[cash_col]
        idx = s.index[s.values < 0]
        return float(idx.min()) if len(idx) else np.nan

    def _first_sustained_ge_zero(g: pd.DataFrame, k: int = 3) -> float:
        s = g.set_index(month_col)[cf_col].sort_index()
        ok = (s >= 0).astype(int).rolling(k, min_periods=k).sum() == k
        idx = ok[ok].index
        return float(idx.min()) if len(idx) else np.nan

    rows = []
    for (env, strat, sim), g in df.sort_values([env_col, strat_col, sim_col, month_col]).groupby([env_col, strat_col, sim_col]):
        rows.append({
            env_col: env,
            strat_col: strat,
            sim_col: sim,
            "t_insolvency": _first_cash_negative(g),
            "t_breakeven": _first_sustained_ge_zero(g, k=breakeven_k),
            "min_cash": float(g[cash_col].min()),
        })
    timings = pd.DataFrame(rows, columns=[env_col, strat_col, sim_col, "t_insolvency", "t_breakeven", "min_cash"])

    T = int(df[month_col].max())
    
    # Survival
    surv = (timings.assign(neg=lambda d: d["min_cash"] < 0)
                    .groupby([env_col, strat_col])["neg"].mean()
                    .reset_index(name="prob_insolvent_by_T"))
    surv["survival_prob"] = 1.0 - surv["prob_insolvent_by_T"]

    # Cash quantiles
    last = df[df[month_col] == T]
    cash_q = (last.groupby([env_col, strat_col])[cash_col]
                  .quantile([0.10, 0.50, 0.90]).unstack().reset_index()
                  .rename(columns={0.10:"cash_q10", 0.50:"cash_med", 0.90:"cash_q90"}))

    # DSCR
    m12 = 12 if T >= 12 else T
    if "dscr" in df.columns:
        dscr_q = (df[df[month_col] == m12]
                    .groupby([env_col, strat_col])["dscr"]
                    .quantile([0.10, 0.50, 0.90]).unstack().reset_index()
                    .rename(columns={0.10:"dscr_q10", 0.50:"dscr_med", 0.90:"dscr_q90"}))
    else:
        dscr_q = pd.DataFrame({env_col: [], strat_col: [], "dscr_q10": [], "dscr_med": [], "dscr_q90": []})

    def _med_or_nan(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.median()) if len(s) else np.nan

    tim_summary = (timings.groupby([env_col, strat_col]).agg(
        median_time_to_insolvency_months=("t_insolvency", _med_or_nan),
        median_time_to_breakeven_months=("t_breakeven", _med_or_nan),
    ).reset_index())

    matrix_row = (surv[[env_col, strat_col, "survival_prob"]]
                    .merge(cash_q, on=[env_col, strat_col], how="left")
                    .merge(dscr_q, on=[env_col, strat_col], how="left")
                    .merge(tim_summary, on=[env_col, strat_col], how="left"))

    return matrix_row.iloc[0].to_dict(), timings


def calculate_business_viability_score(kpis: dict) -> tuple:
    """
    Calculate a business viability score based on KPIs.
    
    Args:
        kpis: Dictionary of KPIs
    
    Returns:
        Tuple of (score, interpretation)
    """
    score = 0
    max_score = 100
    
    # Survival probability (40% of score)
    survival_prob = kpis.get('survival_prob', 0)
    score += survival_prob * 40
    
    # Cash position (30% of score)
    cash_med = kpis.get('cash_med', 0)
    if cash_med > 50000:
        score += 30
    elif cash_med > 25000:
        score += 20
    elif cash_med > 0:
        score += 10
    
    # DSCR (30% of score)
    dscr_med = kpis.get('dscr_med', 0)
    if dscr_med > 1.5:
        score += 30
    elif dscr_med > 1.25:
        score += 20
    elif dscr_med > 1.0:
        score += 10
    
    # Interpretation
    if score >= 80:
        interpretation = "Strong: Low risk, attractive to lenders"
    elif score >= 60:
        interpretation = "Good: Moderate risk, likely fundable"
    elif score >= 40:
        interpretation = "Fair: Higher risk, may need adjustments"
    else:
        interpretation = "Weak: High risk, significant concerns"
    
    return score, interpretation


def compute_sensitivity_analysis(results_df: pd.DataFrame, param_name: str) -> dict:
    """
    Compute sensitivity analysis for a given parameter.
    
    Args:
        results_df: DataFrame with simulation results
        param_name: Name of parameter to analyze
    
    Returns:
        Dictionary with sensitivity metrics
    """
    if param_name not in results_df.columns:
        return {"error": f"Parameter {param_name} not found in results"}
    
    # Group by parameter value and compute key metrics
    grouped = results_df.groupby(param_name).agg({
        'cash_balance': ['mean', 'std'],
        'active_members': 'mean',
        'net_cash_flow': 'mean'
    }).round(2)
    
    return {
        "parameter": param_name,
        "analysis": grouped.to_dict(),
        "interpretation": f"Sensitivity analysis for {param_name}"
    }