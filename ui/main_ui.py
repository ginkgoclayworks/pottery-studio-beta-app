#!/usr/bin/env python3
"""
Main Streamlit application for the pottery studio simulator.
Orchestrates the UI components and simulation execution.
"""

import io
import json
import zipfile
import streamlit as st
import pandas as pd

# Import refactored modules
from config.parameters import CONSOLIDATED_GROUPS
from config.scenarios import SCENARIOS, STRATEGIES, DEFAULT_CAPEX_ITEMS
from simulation.engine import run_cell_cached
from simulation.validation import preflight_validate, normalize_capex_items
from ui.components import (
    render_parameter_group, render_loan_controls, render_simulation_settings
)
from analysis.metrics import compute_kpis_from_cell, summarize_cell
from visualization.charts import apply_visualization_patches


def initialize_streamlit():
    """Initialize Streamlit configuration and page setup."""
    st.set_page_config(page_title="GCWS Simulator - Enhanced", layout="wide")
    st.title("Ginkgo Clayworks — Scenario Explorer (Enhanced)")
    st.caption("Enhanced app with consolidated parameters and improved visualization")


def render_sidebar():
    """
    Render the sidebar with all configuration controls.
    
    Returns:
        tuple: (env, strat, loan_settings, sim_settings)
    """
    with st.sidebar:
        with st.expander("About this model", expanded=False):
            st.markdown("Enhanced version with consolidated parameters and improved visualization.")
        
        st.header("Configuration")
        st.caption("Hover over any label for explanation. Colors indicate how likely parameters are to vary between studios.")
        st.session_state["_show_hints"] = st.toggle("Show range hints", value=True)

        # Scenario and strategy selection
        scen_names = [s["name"] for s in SCENARIOS]
        strat_names = [s["name"] for s in STRATEGIES]

        scen_sel = st.selectbox("Scenario preset", scen_names, index=0)
        strat_sel = st.selectbox("Strategy preset", strat_names, index=0)

        # Loan controls
        loan_settings = render_loan_controls()
        
        # Simulation settings
        sim_settings = render_simulation_settings()

        # Get selected presets
        env = next(s for s in SCENARIOS if s["name"] == scen_sel)
        strat = next(s for s in STRATEGIES if s["name"] == strat_sel)
        strat["N_SIMULATIONS"] = sim_settings["N_SIMULATIONS"]
        
        # Render consolidated parameter groups
        for group_name, group_config in CONSOLIDATED_GROUPS.items():
            with st.expander(group_config['title'], expanded=(group_name == 'business_core')):
                if group_name in ['business_core', 'market_response']:
                    env = render_parameter_group(group_name, group_config, env, "env")
                else:
                    strat = render_parameter_group(group_name, group_config, strat, "strat")

        # Equipment section
        render_equipment_section(strat)
        
    return env, strat, loan_settings, sim_settings


def render_equipment_section(strat):
    """
    Render the equipment configuration section.
    
    Args:
        strat: Strategy dictionary to update with equipment items
    """
    with st.expander("Equipment", expanded=True):
        st.markdown("**Staged purchases (all equipment)**")
        
        capex_existing = strat.get("CAPEX_ITEMS", [])
        capex_df_default = pd.DataFrame(capex_existing) if capex_existing else pd.DataFrame(DEFAULT_CAPEX_ITEMS)
        
        capex_df = st.data_editor(
            capex_df_default,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Include"),
                "label": st.column_config.TextColumn("Item"),
                "count": st.column_config.NumberColumn("Count", min_value=1, step=1),
                "unit_cost": st.column_config.NumberColumn("Unit cost ($)", min_value=0, step=10),
                "month": st.column_config.NumberColumn("Trigger month", min_value=0, step=1),
                "member_threshold": st.column_config.NumberColumn("Trigger members", min_value=0, step=1),
                "finance_504": st.column_config.CheckboxColumn("Finance via 504"),
            },
        )
        
        strat["CAPEX_ITEMS"] = normalize_capex_items(capex_df)


def render_results(env, strat, sim_settings):
    """
    Render simulation results and visualizations.
    
    Args:
        env: Environment configuration
        strat: Strategy configuration
        sim_settings: Simulation settings
    """
    # Clear cache and run simulation
    try:
        run_cell_cached.clear()
        st.cache_data.clear()
    except Exception:
        pass
    
    # Apply visualization patches
    apply_visualization_patches()
    
    with st.spinner("Running simulator…"):
        cache_key = f"v7|{json.dumps(env, sort_keys=True)}|{json.dumps(strat, sort_keys=True)}|{sim_settings['RANDOM_SEED']}"
        df_cell, eff, images, manifest = run_cell_cached(env, strat, sim_settings["RANDOM_SEED"], cache_key)
        st.session_state["df_result"] = df_cell
        
    st.subheader(f"Results — {env['name']} | {strat['name']}")

    # Display KPIs
    kpi_cell = compute_kpis_from_cell(df_cell)
    row_dict, _tim = summarize_cell(df_cell)
    
    # Show metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Survival prob", f"{kpi_cell.get('survival_prob', 0):.2f}")
    col2.metric("Cash p50 ($k)", f"{(kpi_cell.get('cash_med', 0)/1e3):,.0f}")
    col3.metric("DSCR p50", f"{kpi_cell.get('dscr_med', 0):.2f}")
    col4.metric("Breakeven (mo)", f"{row_dict.get('median_time_to_breakeven_months', 0):.0f}")
    
    # Display charts
    st.markdown("#### Captured charts")
    for fname, data in images:
        st.image(data, caption=fname, use_container_width=True)

    # Download bundle
    render_download_section(env, strat, images, manifest, df_cell)


def render_download_section(env, strat, images, manifest, df_cell):
    """
    Render the download section with plots and raw data.
    
    Args:
        env: Environment configuration
        strat: Strategy configuration
        images: List of image tuples
        manifest: Image manifest
        df_cell: Results DataFrame
    """
    # Create downloadable zip of plots
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for fname, data in images:
            zf.writestr(fname, data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download plots (zip)", 
            data=buf.getvalue(),
            file_name=f"{env['name']}__{strat['name']}_plots.zip"
        )
    
    with col2:
        csv_buffer = io.StringIO()
        df_cell.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"{env['name']}__{strat['name']}_results.csv",
            mime="text/csv"
        )

    # Show raw results preview
    st.markdown("#### Raw results")
    st.dataframe(df_cell.head(250))


def main():
    """Main application entry point."""
    initialize_streamlit()
    
    # Render sidebar and get configurations
    env, strat, loan_settings, sim_settings = render_sidebar()
    
    # Store loan settings in session state
    for key, value in loan_settings.items():
        st.session_state[key] = value
    
    # Preflight validation
    if not preflight_validate(strat):
        st.stop()

    # Main content area
    run_btn = st.button("Run simulation")

    if run_btn:
        render_results(env, strat, sim_settings)


if __name__ == "__main__":
    main()
