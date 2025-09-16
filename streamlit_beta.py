#!/usr/bin/env python3
"""
Enhanced Advanced Analysis App - Built on your app.py foundation
Fixes redundancies, improves visualizations, uses your actual defaults
"""

import io, json, re, zipfile
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Your existing imports
from modular_simulator import get_default_cfg
from final_batch_adapter import run_original_once
from sba_export import export_to_sba_workbook

# Page configuration
st.set_page_config(
    page_title="Pottery Studio Business Simulator - Advanced",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONSOLIDATE REDUNDANT PARAMETERS ====================

# FIXED: Consolidate rent parameters (was RENT vs RENT_SCENARIOS)
CONSOLIDATED_PARAM_SPECS = {
    # Business Fundamentals (GREEN - most likely to change)
    "MONTHLY_RENT": {"type": "int", "min": 1000, "max": 10000, "step": 50, "label": "Monthly Rent ($)", 
                     "color": "green", "desc": "Monthly base rent for the space", "rec": (2500, 5500)},
    "MEMBERSHIP_PRICE": {"type": "int", "min": 100, "max": 300, "step": 5, "label": "Monthly Membership ($)", 
                         "color": "green", "desc": "Monthly membership fee charged to members", "rec": (120, 220)},
    "MAX_MEMBERS": {"type": "int", "min": 30, "max": 300, "step": 10, "label": "Studio Capacity", 
                    "color": "green", "desc": "Maximum members studio can accommodate", "rec": (70, 110)},
    "OWNER_COMPENSATION": {"type": "int", "min": 0, "max": 5000, "step": 50, "label": "Owner Monthly Draw ($)", 
                          "color": "green", "desc": "Monthly income you take from business", "rec": (0, 1500)},
    
    # Market Response (AMBER - may need adjustment)
    "REFERENCE_PRICE": {"type": "int", "min": 50, "max": 250, "step": 5, "label": "Competitor Average Price ($)", 
                        "color": "amber", "desc": "What similar studios charge for monthly membership", "rec": (80, 180)},
    "JOIN_PRICE_ELASTICITY": {"type": "float", "min": -3.0, "max": 0.0, "step": 0.05, "label": "Price Sensitivity - New Members", 
                              "color": "amber", "desc": "How sensitive potential members are to pricing", "rec": (-2.0, -1.0)},
    "CHURN_PRICE_ELASTICITY": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "label": "Price Sensitivity - Retention", 
                               "color": "amber", "desc": "How pricing affects member retention", "rec": (0.8, 1.4)},
    
    # Equipment (GREEN - varies significantly between studios)  
    "POTTERY_WHEELS": {"type": "int", "min": 1, "max": 20, "step": 1, "label": "Pottery Wheels", 
                       "color": "green", "desc": "Number of wheels at opening", "rec": (6, 12)},
    "WHEEL_COST": {"type": "int", "min": 800, "max": 3500, "step": 100, "label": "Cost per Wheel ($)", 
                   "color": "green", "desc": "Purchase price per pottery wheel", "rec": (1200, 2500)},
    "DRYING_RACKS": {"type": "int", "min": 1, "max": 25, "step": 1, "label": "Drying Racks", 
                     "color": "green", "desc": "Number of ware racks", "rec": (8, 15)},
    
    # Revenue Streams (AMBER - business model dependent)
    "WORKSHOPS_ENABLED": {"type": "bool", "label": "Enable Workshops", "color": "amber", 
                          "desc": "Short pottery experiences for beginners", "default": True},
    "WORKSHOP_FREQUENCY": {"type": "float", "min": 0.0, "max": 12.0, "step": 0.5, "label": "Workshops per Month", 
                           "color": "amber", "desc": "Average monthly workshops", "rec": (1, 4)},
    "WORKSHOP_PRICE": {"type": "float", "min": 25.0, "max": 150.0, "step": 5.0, "label": "Workshop Price ($)", 
                       "color": "amber", "desc": "Price per workshop attendee", "rec": (60, 100)},
    "WORKSHOP_CAPACITY": {"type": "int", "min": 4, "max": 20, "step": 1, "label": "Workshop Capacity", 
                          "color": "amber", "desc": "Average attendees per workshop", "rec": (8, 15)},
    
    # Classes (AMBER - varies by teaching focus)
    "CLASSES_ENABLED": {"type": "bool", "label": "Enable Classes", "color": "amber", 
                        "desc": "Multi-week pottery courses", "default": True},
    "CLASS_SCHEDULE_MODE": {"type": "select", "options": ["monthly", "semester"], "label": "Class Schedule", 
                            "color": "amber", "desc": "Monthly ongoing vs semester terms", "default": "semester"},
    "CLASS_COHORTS": {"type": "int", "min": 0, "max": 8, "step": 1, "label": "Classes per Period", 
                      "color": "amber", "desc": "New classes started per period", "rec": (1, 3)},
    "CLASS_SIZE": {"type": "int", "min": 4, "max": 16, "step": 1, "label": "Class Size", 
                   "color": "amber", "desc": "Students per class", "rec": (6, 12)},
    "CLASS_PRICE": {"type": "int", "min": 100, "max": 600, "step": 25, "label": "Class Series Price ($)", 
                    "color": "amber", "desc": "Price for full multi-week course", "rec": (200, 400)},
    
    # Economic Environment (RED - rarely changed)
    "ECONOMIC_STRESS_LEVEL": {"type": "select", 
                               "options": [("Normal", 0.05), ("Moderate", 0.08), ("Uncertain", 0.12), ("Stressed", 0.18), ("Recession", 0.30)], 
                               "label": "Economic Stress Level", "color": "red", 
                               "desc": "How often economic stress affects business", "default": ("Moderate", 0.08)},
    "DOWNTURN_JOIN_IMPACT": {"type": "float", "min": 0.3, "max": 1.0, "step": 0.05, "label": "Economic Impact on Joins", 
                             "color": "red", "desc": "Join rate multiplier during stress", "rec": (0.6, 0.9)},
    "DOWNTURN_CHURN_IMPACT": {"type": "float", "min": 1.0, "max": 2.5, "step": 0.05, "label": "Economic Impact on Churn", 
                              "color": "red", "desc": "Churn rate multiplier during stress", "rec": (1.1, 1.6)},
    
    # Financing (RED - set once during planning)
    "LOAN_504_AMOUNT": {"type": "int", "min": 0, "max": 600000, "step": 10000, "label": "SBA 504 Loan ($)", 
                        "color": "red", "desc": "Equipment and real estate loan", "rec": (200000, 400000)},
    "LOAN_504_RATE": {"type": "float", "min": 0.04, "max": 0.12, "step": 0.0025, "label": "504 Interest Rate (%)", 
                      "color": "red", "desc": "Annual percentage rate", "rec": (0.06, 0.08)},
    "LOAN_7A_AMOUNT": {"type": "int", "min": 0, "max": 300000, "step": 5000, "label": "SBA 7(a) Loan ($)", 
                       "color": "red", "desc": "Working capital loan", "rec": (75000, 150000)},
    "LOAN_7A_RATE": {"type": "float", "min": 0.05, "max": 0.15, "step": 0.0025, "label": "7(a) Interest Rate (%)", 
                     "color": "red", "desc": "Annual percentage rate", "rec": (0.07, 0.10)},
}

# ==================== PARAMETER GROUPINGS ====================

PARAMETER_GROUPS = {
    "business_fundamentals": {
        "title": "Business Fundamentals",
        "description": "Core operational parameters that define your studio",
        "color": "green",
        "basic": ["MONTHLY_RENT", "MEMBERSHIP_PRICE", "MAX_MEMBERS", "OWNER_COMPENSATION"],
        "detailed": []
    },
    "market_pricing": {
        "title": "Market & Pricing Response", 
        "description": "How your studio responds to market conditions and pricing",
        "color": "amber",
        "basic": ["REFERENCE_PRICE"],
        "detailed": ["JOIN_PRICE_ELASTICITY", "CHURN_PRICE_ELASTICITY"]
    },
    "equipment": {
        "title": "Equipment & Infrastructure",
        "description": "Studio equipment and setup costs",
        "color": "green", 
        "basic": ["POTTERY_WHEELS", "DRYING_RACKS"],
        "detailed": ["WHEEL_COST"]
    },
    "revenue_workshops": {
        "title": "Workshop Revenue Stream",
        "description": "Short pottery experiences and beginner programs",
        "color": "amber",
        "basic": ["WORKSHOPS_ENABLED", "WORKSHOP_FREQUENCY", "WORKSHOP_PRICE"],
        "detailed": ["WORKSHOP_CAPACITY"]
    },
    "revenue_classes": {
        "title": "Class Revenue Stream", 
        "description": "Multi-week structured pottery courses",
        "color": "amber",
        "basic": ["CLASSES_ENABLED", "CLASS_SCHEDULE_MODE", "CLASS_COHORTS"],
        "detailed": ["CLASS_SIZE", "CLASS_PRICE"]
    },
    "economic_environment": {
        "title": "Economic Environment",
        "description": "External economic conditions affecting your studio",
        "color": "red",
        "basic": ["ECONOMIC_STRESS_LEVEL"],
        "detailed": ["DOWNTURN_JOIN_IMPACT", "DOWNTURN_CHURN_IMPACT"]
    },
    "financing": {
        "title": "SBA Loan Financing",
        "description": "Loan terms and financing structure", 
        "color": "red",
        "basic": ["LOAN_504_AMOUNT", "LOAN_7A_AMOUNT"],
        "detailed": ["LOAN_504_RATE", "LOAN_7A_RATE"]
    }
}

# ==================== IMPROVED VISUALIZATION FUNCTIONS ====================

def create_enhanced_cash_flow_chart(df_results):
    """Create enhanced cash flow visualization with multiple perspectives"""
    
    # Get cash flow data by month
    cash_by_month = df_results.groupby('month')['cash_balance'].agg(['mean', 'median', 'std']).reset_index()
    cash_percentiles = df_results.groupby('month')['cash_balance'].quantile([0.1, 0.25, 0.75, 0.9]).unstack()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cash Flow Distribution Over Time', 'Cash Flow Risk Analysis', 
                       'Monthly Cash Flow Changes', 'Survival Probability'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Main cash flow chart with confidence bands
    fig.add_trace(
        go.Scatter(x=cash_by_month['month'], y=cash_percentiles[0.9], 
                  fill=None, mode='lines', line_color='rgba(0,100,80,0)', showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=cash_by_month['month'], y=cash_percentiles[0.1],
                  fill='tonexty', mode='lines', line_color='rgba(0,100,80,0)',
                  name='10th-90th percentile', fillcolor='rgba(0,100,80,0.2)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=cash_by_month['month'], y=cash_by_month['median'],
                  mode='lines', name='Median', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Risk analysis - probability of negative cash
    risk_by_month = df_results.groupby('month').apply(
        lambda x: (x['cash_balance'] < 0).mean()
    ).reset_index(name='insolvency_prob')
    
    fig.add_trace(
        go.Scatter(x=risk_by_month['month'], y=risk_by_month['insolvency_prob'],
                  mode='lines+markers', name='Insolvency Risk', 
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Monthly changes
    cash_changes = df_results.groupby(['simulation_id', 'month'])['cash_balance'].first().groupby('simulation_id').diff()
    change_stats = cash_changes.groupby(level=1).agg(['mean', 'std']).droplevel(0, axis=1)
    
    fig.add_trace(
        go.Scatter(x=change_stats.index, y=change_stats['mean'],
                  mode='lines', name='Avg Monthly Change', 
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    
    # Survival probability over time
    survival_by_month = df_results.groupby('month').apply(
        lambda x: (x.groupby('simulation_id')['cash_balance'].min() >= 0).mean()
    ).reset_index(name='survival_prob')
    
    fig.add_trace(
        go.Scatter(x=survival_by_month['month'], y=survival_by_month['survival_prob'],
                  mode='lines+markers', name='Survival Rate',
                  line=dict(color='darkblue', width=3)),
        row=2, col=2
    )
    
    # Add insolvency line to main chart
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                 annotation_text="Insolvency Line", row=1, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Enhanced Cash Flow Analysis",
        title_x=0.5
    )
    
    return fig

def create_business_model_comparison(df_results):
    """Compare different business model metrics"""
    
    # Calculate key business metrics
    final_month = df_results['month'].max()
    
    # Revenue breakdown by source (if available)
    revenue_cols = [col for col in df_results.columns if 'revenue' in col.lower()]
    
    if revenue_cols:
        revenue_data = []
        for col in revenue_cols:
            total_revenue = df_results[df_results['month'] == final_month][col].sum()
            revenue_data.append({
                'source': col.replace('revenue_', '').replace('_', ' ').title(),
                'amount': total_revenue
            })
        
        revenue_df = pd.DataFrame(revenue_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue by Source', 'Member Growth Trajectory'),
            specs=[[{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Revenue pie chart
        fig.add_trace(
            go.Pie(labels=revenue_df['source'], values=revenue_df['amount'],
                  name="Revenue Sources"),
            row=1, col=1
        )
        
        # Member growth if available
        if 'active_members' in df_results.columns:
            member_growth = df_results.groupby('month')['active_members'].median()
            fig.add_trace(
                go.Scatter(x=member_growth.index, y=member_growth.values,
                          mode='lines+markers', name='Median Members'),
                row=1, col=2
            )
    
    return fig

def create_risk_dashboard(df_results):
    """Create comprehensive risk analysis dashboard"""
    
    # Calculate various risk metrics
    monthly_risks = df_results.groupby('month').agg({
        'cash_balance': ['mean', 'std', lambda x: (x < 0).mean(), lambda x: (x < 10000).mean()],
        'dscr': ['mean', 'std', lambda x: (x < 1.25).mean()] if 'dscr' in df_results.columns else [np.nan, np.nan, np.nan]
    }).round(3)
    
    # Flatten column names
    monthly_risks.columns = ['_'.join(col).strip() for col in monthly_risks.columns]
    monthly_risks = monthly_risks.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cash Risk Over Time', 'DSCR Risk Profile', 
                       'Risk Correlation Matrix', 'Monte Carlo Risk Distribution'),
    )
    
    # Cash risk metrics
    if 'cash_balance_<lambda_0>' in monthly_risks.columns:
        fig.add_trace(
            go.Scatter(x=monthly_risks['month'], 
                      y=monthly_risks['cash_balance_<lambda_0>'] * 100,
                      mode='lines+markers', name='Prob(Cash < $0)', 
                      line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=monthly_risks['month'], 
                      y=monthly_risks['cash_balance_<lambda_1>'] * 100,
                      mode='lines+markers', name='Prob(Cash < $10k)', 
                      line=dict(color='orange')),
            row=1, col=1
        )
    
    # DSCR risk if available
    if 'dscr' in df_results.columns and 'dscr_<lambda>' in monthly_risks.columns:
        fig.add_trace(
            go.Scatter(x=monthly_risks['month'], 
                      y=monthly_risks['dscr_<lambda>'] * 100,
                      mode='lines+markers', name='Prob(DSCR < 1.25)', 
                      line=dict(color='purple')),
            row=1, col=2
        )
    
    # Risk distribution at final month
    final_month_data = df_results[df_results['month'] == df_results['month'].max()]
    
    fig.add_trace(
        go.Histogram(x=final_month_data['cash_balance'], 
                    name='Final Cash Distribution',
                    nbinsx=30),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Risk Analysis Dashboard",
        title_x=0.5
    )
    
    return fig

# ==================== ENHANCED PARAMETER RENDERING ====================

def render_parameter_group(group_name, group_config, params_state):
    """Render a parameter group with progressive disclosure"""
    
    st.subheader(f"{group_config['title']}")
    st.write(group_config['description'])
    
    # Color coding indicator
    color_map = {"green": "🟢", "amber": "🟡", "red": "🔴"}
    color_name_map = {"green": "Most likely to vary", "amber": "May need adjustment", "red": "Rarely changed"}
    
    st.caption(f"{color_map[group_config['color']]} {color_name_map[group_config['color']]}")
    
    # Always show basic parameters
    for param_name in group_config['basic']:
        param_spec = CONSOLIDATED_PARAM_SPECS[param_name]
        params_state[param_name] = render_single_parameter(param_name, param_spec, params_state.get(param_name))
    
    # Detailed parameters in expander
    if group_config['detailed']:
        with st.expander("🔧 Detailed Settings", expanded=False):
            for param_name in group_config['detailed']:
                param_spec = CONSOLIDATED_PARAM_SPECS[param_name]
                params_state[param_name] = render_single_parameter(param_name, param_spec, params_state.get(param_name))
    
    return params_state

def render_single_parameter(param_name, param_spec, current_value):
    """Render a single parameter with appropriate widget"""
    
    param_type = param_spec['type']
    label = param_spec['label']
    help_text = build_help_text(param_spec)
    
    # Set default value if not provided
    if current_value is None:
        current_value = param_spec.get('default', get_default_value(param_spec))
    
    # Render appropriate widget
    if param_type == 'bool':
        return st.checkbox(label, value=current_value, help=help_text)
    
    elif param_type == 'int':
        return st.slider(
            label, 
            min_value=param_spec['min'], 
            max_value=param_spec['max'], 
            value=current_value,
            step=param_spec['step'],
            help=help_text
        )
    
    elif param_type == 'float':
        return st.slider(
            label,
            min_value=param_spec['min'],
            max_value=param_spec['max'], 
            value=current_value,
            step=param_spec['step'],
            help=help_text
        )
    
    elif param_type == 'select':
        options = param_spec['options']
        if isinstance(options[0], tuple):
            # Options with values (like economic stress levels)
            option_labels = [opt[0] for opt in options]
            option_values = [opt[1] for opt in options]
            
            # Find current selection
            try:
                current_index = option_values.index(current_value[1] if isinstance(current_value, tuple) else current_value)
            except (ValueError, TypeError):
                current_index = 0
                
            selected = st.selectbox(
                label,
                options=options,
                index=current_index,
                format_func=lambda x: x[0] if isinstance(x, tuple) else str(x),
                help=help_text
            )
            return selected
        else:
            # Simple string options
            current_index = options.index(current_value) if current_value in options else 0
            return st.selectbox(label, options=options, index=current_index, help=help_text)
    
    return current_value

def build_help_text(param_spec):
    """Build comprehensive help text from parameter specification"""
    parts = []
    
    if 'desc' in param_spec:
        parts.append(param_spec['desc'])
    
    if 'rec' in param_spec and isinstance(param_spec['rec'], (list, tuple)) and len(param_spec['rec']) == 2:
        parts.append(f"Recommended range: {param_spec['rec'][0]} - {param_spec['rec'][1]}")
    
    return " | ".join(parts)

def get_default_value(param_spec):
    """Get sensible default value from parameter specification"""
    if param_spec['type'] == 'bool':
        return False
    elif param_spec['type'] in ['int', 'float']:
        return param_spec['min']
    elif param_spec['type'] == 'select':
        return param_spec['options'][0]
    return None

# ==================== MAIN APPLICATION ====================

def main():
    st.title("🏺 Pottery Studio Business Simulator - Advanced Analysis")
    st.write("**Complete parameter control with enhanced visualization and risk analysis**")
    
    # Initialize session state for parameters
    if 'params_state' not in st.session_state:
        st.session_state.params_state = {}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Studio Configuration")
        
        # Render all parameter groups
        for group_name, group_config in PARAMETER_GROUPS.items():
            with st.expander(f"{group_config['title']}", expanded=(group_name == 'business_fundamentals')):
                st.session_state.params_state = render_parameter_group(
                    group_name, group_config, st.session_state.params_state
                )
        
        # Simulation controls
        st.subheader("Simulation Controls")
        num_simulations = st.slider("Number of Simulations", 25, 500, 100, 25,
                                   help="More simulations = more accurate results but slower")
        time_horizon = st.selectbox("Time Horizon", [12, 24, 36, 48, 60], index=2,
                                   format_func=lambda x: f"{x} months ({x//12} years)")
        random_seed = st.number_input("Random Seed", 0, 1000000, 42, 1,
                                     help="Fix for reproducible results")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col2:
        if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
            with st.spinner("Running advanced simulation analysis..."):
                # Convert parameters to format expected by your simulator
                simulator_params = convert_params_for_simulator(st.session_state.params_state)
                simulator_params.update({
                    'N_SIMULATIONS': num_simulations,
                    'MONTHS': time_horizon,
                    'RANDOM_SEED': random_seed
                })
                
                try:
                    # Run simulation using your existing function
                    results = run_original_once("modular_simulator.py", simulator_params)
                    df_results, effective_config = (results if isinstance(results, tuple) else (results, None))
                    
                    if df_results is not None and not df_results.empty:
                        st.session_state.simulation_results = df_results
                        st.session_state.effective_config = effective_config
                        st.success("Advanced analysis completed!")
                    else:
                        st.error("Simulation returned empty results")
                        
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if 'simulation_results' in st.session_state:
        df_results = st.session_state.simulation_results
        
        # Enhanced metrics display
        st.subheader("Business Performance Summary")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        final_month = df_results['month'].max()
        final_data = df_results[df_results['month'] == final_month]
        
        with col1:
            survival_rate = (final_data['cash_balance'] > 0).mean()
            st.metric("Survival Rate", f"{survival_rate:.1%}")
            
            median_revenue = df_results['revenue_total'].median() if 'revenue_total' in df_results.columns else 0
            st.metric("Median Monthly Revenue", f"${median_revenue:,.0f}")
        
        with col2:
            median_cash = final_data['cash_balance'].median()
            st.metric("Final Cash (Median)", f"${median_cash:,.0f}")
            
            if 'active_members' in df_results.columns:
                final_members = final_data['active_members'].median()
                st.metric("Final Members (Median)", f"{final_members:.0f}")
        
        with col3:
            cash_10th = final_data['cash_balance'].quantile(0.1)
            st.metric("Cash 10th Percentile", f"${cash_10th:,.0f}")
            
            if 'dscr' in df_results.columns:
                median_dscr = df_results[df_results['month'] == min(12, final_month)]['dscr'].median()
                st.metric("DSCR @ Year 1", f"{median_dscr:.2f}")
        
        with col4:
            # Calculate break-even month
            if 'operating_profit' in df_results.columns:
                breakeven_data = df_results.groupby('simulation_id')['operating_profit'].apply(
                    lambda x: x.index[x > 0].min() if (x > 0).any() else np.nan
                )
                median_breakeven = breakeven_data.median()
                st.metric("Median Break-even", f"Month {median_breakeven:.0f}" if not pd.isna(median_breakeven) else "N/A")
        
        # Enhanced visualizations
        st.subheader("Enhanced Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Cash Flow Analysis", "Business Model Metrics", "Risk Dashboard"])
        
        with tab1:
            st.plotly_chart(create_enhanced_cash_flow_chart(df_results), use_container_width=True)
        
        with tab2:
            st.plotly_chart(create_business_model_comparison(df_results), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_risk_dashboard(df_results), use_container_width=True)
        
        # Data export options
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Summary CSV"):
                summary_data = calculate_summary_metrics(df_results)
                csv = pd.DataFrame([summary_data]).to_csv(index=False)
                st.download_button(
                    "Click to Download Summary",
                    data=csv,
                    file_name="studio_analysis_summary.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Full Results CSV"):
                csv = df_results.to_csv(index=False)
                st.download_button(
                    "Click to Download Full Data", 
                    data=csv,
                    file_name="studio_simulation_detailed.csv",
                    mime="text/csv"
                )

def convert_params_for_simulator(params_state):
    """Convert consolidated parameters back to format expected by your simulator"""
    
    # Map consolidated parameters back to original parameter names
    simulator_params = {}
    
    # Handle basic mappings
    param_mapping = {
        'MONTHLY_RENT': 'RENT',
        'MEMBERSHIP_PRICE': 'PRICE', 
        'MAX_MEMBERS': 'MEMBER_CAP',
        'OWNER_COMPENSATION': 'OWNER_DRAW',
        'POTTERY_WHEELS': 'N_WHEELS_START',
        'DRYING_RACKS': 'N_RACKS_START',
        'WORKSHOP_FREQUENCY': 'WORKSHOPS_PER_MONTH',
        'WORKSHOP_PRICE': 'WORKSHOP_FEE',
        'WORKSHOP_CAPACITY': 'WORKSHOP_AVG_ATTENDANCE',
        'CLASS_COHORTS': 'CLASS_COHORTS_PER_MONTH',
        'CLASS_SIZE': 'CLASS_CAP_PER_COHORT',
        'CLASS_PRICE': 'CLASS_PRICE',
        'LOAN_504_AMOUNT': 'LOAN_OVERRIDE_504',
        'LOAN_7A_AMOUNT': 'LOAN_OVERRIDE_7A',
        'LOAN_504_RATE': 'LOAN_504_ANNUAL_RATE',
        'LOAN_7A_RATE': 'LOAN_7A_ANNUAL_RATE'
    }
    
    for new_name, old_name in param_mapping.items():
        if new_name in params_state:
            simulator_params[old_name] = params_state[new_name]
    
    # Handle special cases
    if 'ECONOMIC_STRESS_LEVEL' in params_state:
        stress_level = params_state['ECONOMIC_STRESS_LEVEL']
        if isinstance(stress_level, tuple):
            simulator_params['DOWNTURN_PROB_PER_MONTH'] = stress_level[1]
        else:
            simulator_params['DOWNTURN_PROB_PER_MONTH'] = stress_level
    
    if 'DOWNTURN_JOIN_IMPACT' in params_state:
        simulator_params['DOWNTURN_JOIN_MULT'] = params_state['DOWNTURN_JOIN_IMPACT']
    
    if 'DOWNTURN_CHURN_IMPACT' in params_state:
        simulator_params['DOWNTURN_CHURN_MULT'] = params_state['DOWNTURN_CHURN_IMPACT']
    
    if 'CLASS_SCHEDULE_MODE' in params_state:
        if params_state['CLASS_SCHEDULE_MODE'] == 'semester':
            simulator_params['USE_SEMESTER_SCHEDULE'] = True
            simulator_params['CLASSES_PER_SEMESTER'] = params_state.get('CLASS_COHORTS', 2) * 3
        else:
            simulator_params['USE_SEMESTER_SCHEDULE'] = False
    
    # Set required arrays for rent and owner draw
    if 'RENT' in simulator_params:
        simulator_params['RENT_SCENARIOS'] = np.array([float(simulator_params['RENT'])], dtype=float)
    
    if 'OWNER_DRAW' in simulator_params:
        simulator_params['OWNER_DRAW_SCENARIOS'] = [float(simulator_params['OWNER_DRAW'])]
    
    # Enable revenue streams
    simulator_params['WORKSHOPS_ENABLED'] = params_state.get('WORKSHOPS_ENABLED', True)
    simulator_params['CLASSES_ENABLED'] = params_state.get('CLASSES_ENABLED', True)
    
    # Set default market conditions
    simulator_params.update({
        'MARKET_POOLS_INFLOW': {'community_studio': 4, 'home_studio': 2, 'no_access': 3},
        'WOM_RATE': 0.03,
        'LEAD_TO_JOIN_RATE': 0.20,
        'MAX_ONBOARD_PER_MONTH': 10
    })
    
    return simulator_params

def calculate_summary_metrics(df_results):
    """Calculate summary business metrics from simulation results"""
    
    final_month = df_results['month'].max()
    final_data = df_results[df_results['month'] == final_month]
    
    summary = {
        'survival_rate': (final_data['cash_balance'] > 0).mean(),
        'median_final_cash': final_data['cash_balance'].median(),
        'cash_10th_percentile': final_data['cash_balance'].quantile(0.1),
        'cash_90th_percentile': final_data['cash_balance'].quantile(0.9),
    }
    
    # Add revenue metrics if available
    if 'revenue_total' in df_results.columns:
        summary.update({
            'median_monthly_revenue': df_results['revenue_total'].median(),
            'total_revenue_final_year': df_results[df_results['month'] > final_month - 12]['revenue_total'].sum()
        })
    
    # Add member metrics if available  
    if 'active_members' in df_results.columns:
        summary.update({
            'final_member_count': final_data['active_members'].median(),
            'max_member_count': df_results['active_members'].max()
        })
    
    # Add DSCR if available
    if 'dscr' in df_results.columns:
        year_1_dscr = df_results[df_results['month'] == min(12, final_month)]['dscr'].median()
        summary['dscr_year_1'] = year_1_dscr
    
    return summary

if __name__ == "__main__":
    main()