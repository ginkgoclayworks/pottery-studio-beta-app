#!/usr/bin/env python3
"""
Core simulation engine and caching logic.
Handles parameter mapping, simulation execution, and result processing.
"""

import json
from typing import Optional, List, Tuple
import pandas as pd
import streamlit as st
from final_batch_adapter import run_original_once
from config.parameters import consolidate_build_overrides
from visualization.charts import FigureCapture


@st.cache_data(show_spinner=False)
def run_cell_cached(env: dict, strat: dict, seed: int, cache_key: Optional[str] = None):
    """
    Enhanced cached simulation runner with error handling and visualization safety.
    
    Args:
        env: Environment parameters (scenario)
        strat: Strategy parameters
        seed: Random seed for reproducibility
        cache_key: Optional cache key override
    
    Returns:
        tuple: (df_cell, eff, images, manifest)
    """
    if cache_key is None:
        cache_key = f"v7|{json.dumps(env, sort_keys=True)}|{json.dumps(strat, sort_keys=True)}|{seed}"

    # Map consolidated parameters to simulator expectations
    ov = consolidate_build_overrides(env, strat)
    ov["RANDOM_SEED"] = seed

    title_suffix = f"{env['name']} | {strat['name']}"
    
    # Apply visualization safety patches
    _patch_heatmap_calls()
    
    with FigureCapture(title_suffix) as cap:
        try:
            res = run_original_once("modular_simulator.py", ov)
            
            if isinstance(res, tuple):
                df = res[0] if res[0] is not None else pd.DataFrame()
                
                # Validate the dataframe before proceeding
                if df.empty:
                    st.warning("Simulation returned empty DataFrame")
                    return pd.DataFrame(), None, [], []
                    
                # Check for critical columns
                required_cols = ['month', 'cash_balance']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing required columns: {missing_cols}")
                
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)
            return pd.DataFrame(), None, [], []

    df_cell, eff = (res if isinstance(res, tuple) else (res, None))

    if df_cell is None or df_cell.empty:
        st.warning("No data returned from simulation")
        return pd.DataFrame(), None, cap.images, cap.manifest

    df_cell = df_cell.copy()
    df_cell["environment"] = env["name"]
    df_cell["strategy"] = strat["name"]
    if "simulation_id" not in df_cell.columns:
        df_cell["simulation_id"] = 0
        
    return df_cell, eff, cap.images, cap.manifest


def _patch_heatmap_calls():
    """
    Monkey patch to fix heatmap issues - prevents crashes with invalid data.
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    
    original_heatmap = sns.heatmap
    
    def safe_heatmap_wrapper(data, **kwargs):
        # Quick validation
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        if data_array.size == 0 or np.isnan(data_array).all():
            print("Skipping invalid heatmap data")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, 'Invalid heatmap data', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            return plt.gca()
        
        return original_heatmap(data, **kwargs)
    
    sns.heatmap = safe_heatmap_wrapper