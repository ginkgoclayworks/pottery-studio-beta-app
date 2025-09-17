#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated app.py with consolidated parameters - removes redundancies
"""

import io, json, re, zipfile
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from modular_simulator import get_default_cfg
from final_batch_adapter import run_original_once
from sba_export import export_to_sba_workbook
import os
import json

# CONSOLIDATED PARAMETER SPECIFICATIONS - Removes redundancies
CONSOLIDATED_PARAM_SPECS = {
    # Business Fundamentals (GREEN - most likely to vary)
    "MONTHLY_RENT": {"type": "int", "min": 1000, "max": 10_000, "step": 50, "label": "Monthly Rent ($/mo)", 
                     "desc": "Monthly base rent for the space", "rec": (2500, 5500), "color": "green"},
    "RENT_GROWTH_PCT": {"type": "float", "min": 0.0, "max": 15.0, "step": 0.25, "label": "Rent increase per year (%)", 
                        "desc": "Annual rent escalation percentage", "rec": (0.0, 5.0), "color": "red"},
    "MEMBERSHIP_PRICE": {"type": "int", "min": 100, "max": 300, "step": 5, "label": "Membership fee ($/mo)", 
                         "desc": "Monthly membership fee charged to members", "rec": (120, 220), "color": "green"},
    "REFERENCE_PRICE": {"type": "int", "min": 50, "max": 250, "step": 5, "label": "Competitor avg price ($/mo)", 
                        "desc": "What similar studios charge for monthly membership", "rec": (80, 180), "color": "amber"},
    "OWNER_COMPENSATION": {"type": "int", "min": 0, "max": 5000, "step": 50, "label": "Owner draw ($/mo)", 
                          "desc": "Monthly income you take from the business", "rec": (0, 1500), "color": "green"},
    
    # Capacity & Operations (GREEN - varies significantly)
    "STUDIO_CAPACITY": {"type": "int", "min": 30, "max": 300, "step": 10, "label": "Studio capacity (max members)", 
                        "desc": "Maximum members your studio can accommodate", "rec": (70, 110), "color": "green"},
    "EXPANSION_THRESHOLD": {"type": "int", "min": 0, "max": 200, "step": 1, "label": "Expansion threshold (members)", 
                           "desc": "Member count that triggers equipment expansion", "rec": (18, 30), "color": "amber"},
    "MAX_ONBOARD_PER_MONTH": {"type": "int", "min": 1, "max": 200, "step": 1, "label": "Max onboarding / mo", 
                             "desc": "Operational limit on new member onboarding", "rec": (6, 20), "color": "amber"},
    
    # Market Response (AMBER - may need adjustment)
    "JOIN_PRICE_ELASTICITY": {"type": "float", "min": -2.0, "max": 0.0, "step": 0.05, "label": "Join price elasticity", 
                              "desc": "How sensitive potential members are to pricing", "rec": (-2.0, -1.0), "color": "amber"},
    "CHURN_PRICE_ELASTICITY": {"type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "label": "Churn price elasticity", 
                               "desc": "How pricing affects member retention", "rec": (0.8, 1.4), "color": "amber"},
    "WOM_RATE": {"type": "float", "min": 0.0, "max": 0.2, "step": 0.005, "label": "Word-of-mouth rate", 
                 "desc": "Monthly fraction of members who generate qualified leads", "rec": (0.01, 0.06), "color": "amber"},
    "MARKETING_SPEND": {"type": "int", "min": 0, "max": 20_000, "step": 500, "label": "Marketing spend / mo", 
                        "desc": "Monthly paid marketing budget", "rec": (0, 3000), "color": "amber"},
    "CAC": {"type": "int", "min": 50, "max": 2000, "step": 10, "label": "CAC ($/lead)", 
            "desc": "Cost to acquire one qualified lead", "rec": (75, 250), "color": "amber"},
    "LEAD_TO_JOIN_RATE": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "Lead→Join conversion", 
                          "desc": "Share of qualified leads that become members", "rec": (0.10, 0.35), "color": "amber"},
    
    # Economic Environment (RED - rarely changed)
    "ECONOMIC_STRESS_LEVEL": {"type": "select", 
                               "options": [("Normal", 0.05), ("Moderate", 0.08), ("Uncertain", 0.12), ("Stressed", 0.18), ("Recession", 0.30)], 
                               "label": "Economic stress level", "desc": "How often economic stress affects business", 
                               "rec": ("Moderate", 0.08), "color": "red"},
    "DOWNTURN_JOIN_MULT": {"type": "float", "min": 0.2, "max": 1.5, "step": 0.01, "label": "Join multiplier in downturn", 
                           "desc": "Join rate multiplier during economic stress", "rec": (0.6, 1.1), "color": "red"},
    "DOWNTURN_CHURN_MULT": {"type": "float", "min": 0.5, "max": 3.0, "step": 0.05, "label": "Churn multiplier in downturn", 
                            "desc": "Churn rate multiplier during economic stress", "rec": (1.0, 1.8), "color": "red"},
    
    # Market Pools
    "MARKET_POOLS_INFLOW": {"type": "market_inflow", "label": "Market inflow", 
                            "desc": "Monthly counts of potential joiners by segment", "rec": (0, 10), "color": "amber"},
    
    # Workshops (AMBER - business model dependent)
    "WORKSHOPS_ENABLED": {"type": "bool", "label": "Enable workshops", "desc": "Short pottery experiences for beginners", 
                          "default": True, "color": "amber"},
    "WORKSHOPS_PER_MONTH": {"type": "float", "min": 0.0, "max": 12.0, "step": 0.5, "label": "Workshops per month", 
                            "desc": "Average number of workshops per month", "rec": (1, 4), "color": "amber"},
    "WORKSHOP_PRICE": {"type": "float", "min": 15.0, "max": 100.0, "step": 5.0, "label": "Workshop fee per attendee", 
                       "desc": "Price per workshop attendee", "rec": (60, 100), "color": "amber"},
    "WORKSHOP_CAPACITY": {"type": "int", "min": 1, "max": 40, "step": 1, "label": "Avg attendees per workshop", 
                          "desc": "Typical workshop attendance", "rec": (8, 15), "color": "amber"},
    "WORKSHOP_CONV_RATE": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "label": "Workshop conversion rate", 
                           "desc": "Share of attendees who become members", "rec": (0.05, 0.25), "color": "amber"},
    "WORKSHOP_CONV_LAG_MO": {"type": "int", "min": 0, "max": 12, "step": 1, "label": "Conversion lag (months)", 
                             "desc": "Delay between workshop and membership", "rec": (0, 2), "color": "amber"},
    "WORKSHOP_COST_PER_EVENT": {"type": "float", "min": 0.0, "max": 1000.0, "step": 5.0, "label": "Variable cost per workshop", 
                                "desc": "Supplies, instructor, etc.", "rec": (30, 80), "color": "amber"},
    
    # Classes - CONSOLIDATED scheduling parameters
    "CLASSES_ENABLED": {"type": "bool", "label": "Classes enabled", "desc": "Multi-week pottery courses", 
                        "default": True, "color": "amber"},
    "CLASS_SCHEDULE_MODE": {"type": "select", "options": ["monthly", "semester"], "label": "Class schedule type", 
                            "desc": "Monthly ongoing vs semester terms", "default": "semester", "color": "amber"},
    "CLASSES_PER_PERIOD": {"type": "int", "min": 0, "max": 12, "step": 1, "label": "Classes per period", 
                           "desc": "New classes per month (monthly mode) or semester (semester mode)", "rec": (1, 4), "color": "amber"},
    "CLASS_SIZE": {"type": "int", "min": 1, "max": 30, "step": 1, "label": "Class size limit", 
                   "desc": "Maximum students per class", "rec": (6, 14), "color": "amber"},
    "CLASS_PRICE": {"type": "int", "min": 0, "max": 1000, "step": 10, "label": "Class series price", 
                    "desc": "Tuition for full multi-week course", "rec": (200, 600), "color": "amber"},
    "CLASS_CONV_RATE": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "Class conversion rate", 
                        "desc": "Share of students who become members", "rec": (0.05, 0.25), "color": "amber"},
    "CLASS_CONV_LAG_MO": {"type": "int", "min": 0, "max": 12, "step": 1, "label": "Class conv lag (mo)", 
                          "desc": "Delay between class and membership", "rec": (0, 2), "color": "amber"},
    
    # Events (AMBER)
    "BASE_EVENTS_PER_MONTH_LAMBDA": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.5, "label": "Events λ", 
                                     "desc": "Average number of public events per month", "rec": (1, 6), "color": "amber"},
    "EVENTS_MAX_PER_MONTH": {"type": "int", "min": 0, "max": 20, "step": 1, "label": "Events max / mo", 
                             "desc": "Upper bound on monthly events", "rec": (0, 6), "color": "amber"},
    "TICKET_PRICE": {"type": "int", "min": 0, "max": 500, "step": 5, "label": "Ticket price", 
                     "desc": "Price per event attendee", "rec": (55, 110), "color": "amber"},
    
    # Financing - CONSOLIDATED loan parameters
    "LOAN_504_ANNUAL_RATE": {"type": "float", "min": 0.03, "max": 0.20, "step": 0.001, "label": "504 rate (APR)", 
                             "desc": "SBA 504 annual percentage rate", "rec": (0.06, 0.08), "color": "red"},
    "LOAN_504_TERM_YEARS": {"type": "int", "min": 5, "max": 25, "step": 1, "label": "504 term (years)", 
                           "desc": "SBA 504 repayment period", "rec": (15, 20), "color": "red"},
    "IO_MONTHS_504": {"type": "int", "min": 0, "max": 18, "step": 1, "label": "504 interest-only (mo)", 
                      "desc": "Initial interest-only period", "rec": (6, 12), "color": "red"},
    
    "LOAN_7A_ANNUAL_RATE": {"type": "float", "min": 0.05, "max": 0.20, "step": 0.001, "label": "7(a) rate (APR)", 
                           "desc": "SBA 7(a) annual percentage rate", "rec": (0.07, 0.10), "color": "red"},
    "LOAN_7A_TERM_YEARS": {"type": "int", "min": 5, "max": 10, "step": 1, "label": "7(a) term (years)", 
                          "desc": "SBA 7(a) repayment period", "rec": (7, 10), "color": "red"},
    "IO_MONTHS_7A": {"type": "int", "min": 0, "max": 18, "step": 1, "label": "7(a) interest-only (mo)", 
                     "desc": "Initial interest-only period", "rec": (6, 12), "color": "red"},
    
    "LOAN_CONTINGENCY_PCT": {"type": "float", "min": 0.00, "max": 0.25, "step": 0.01, "label": "CapEx contingency (%)", 
                             "desc": "Buffer for equipment cost overruns", "rec": (0.05, 0.15), "color": "red"},
    "RUNWAY_MONTHS": {"type": "int", "min": 0, "max": 24, "step": 1, "label": "Runway months (7a sizing)", 
                      "desc": "Target months of expenses to cover", "rec": (12, 18), "color": "red"},
    "EXTRA_BUFFER": {"type": "int", "min": 0, "max": 20000, "step": 1000, "label": "Extra buffer ($)", 
                     "desc": "Additional working capital buffer", "rec": (10000, 30000), "color": "red"},
    "RESERVE_FLOOR": {"type": "int", "min": 0, "max": 20000, "step": 1000, "label": "Reserve floor ($)", 
                      "desc": "Minimum cash buffer for LOC sizing", "rec": (5000, 15000), "color": "red"},
    
    # Grants (RED)
    "grant_amount": {"type": "int", "min": 0, "max": 100_000, "step": 1000, "label": "Grant amount", 
                     "desc": "One-time grant injection", "rec": (0, 50000), "color": "red"},
    "grant_month": {"type": "int", "min": -1, "max": 36, "step": 1, "label": "Grant month (None=-1)", 
                    "desc": "When grant arrives", "rec": (3, 12), "color": "red"},
}

# CONSOLIDATED PARAMETER GROUPS - Cleaner organization
CONSOLIDATED_GROUPS = {
    "business_core": {
        "title": "Business Fundamentals", 
        "color": "green",
        "basic": ["MONTHLY_RENT", "MEMBERSHIP_PRICE", "STUDIO_CAPACITY", "OWNER_COMPENSATION"],
        "detailed": ["RENT_GROWTH_PCT", "REFERENCE_PRICE"]
    },
    "market_response": {
        "title": "Market & Customer Response", 
        "color": "amber",
        "basic": ["JOIN_PRICE_ELASTICITY", "CHURN_PRICE_ELASTICITY", "MARKET_POOLS_INFLOW"],
        "detailed": ["WOM_RATE", "MARKETING_SPEND", "CAC", "LEAD_TO_JOIN_RATE", "MAX_ONBOARD_PER_MONTH", "EXPANSION_THRESHOLD"]
    },
    "workshops": {
        "title": "Workshop Revenue Stream", 
        "color": "amber",
        "basic": ["WORKSHOPS_ENABLED", "WORKSHOPS_PER_MONTH", "WORKSHOP_PRICE"],
        "detailed": ["WORKSHOP_CAPACITY", "WORKSHOP_CONV_RATE", "WORKSHOP_CONV_LAG_MO", "WORKSHOP_COST_PER_EVENT"]
    },
    "classes": {
        "title": "Class Revenue Stream", 
        "color": "amber",
        "basic": ["CLASSES_ENABLED", "CLASS_SCHEDULE_MODE", "CLASSES_PER_PERIOD"],
        "detailed": ["CLASS_SIZE", "CLASS_PRICE", "CLASS_CONV_RATE", "CLASS_CONV_LAG_MO"]
    },
    "events": {
        "title": "Event Revenue Stream", 
        "color": "amber",
        "basic": ["BASE_EVENTS_PER_MONTH_LAMBDA", "EVENTS_MAX_PER_MONTH", "TICKET_PRICE"],
        "detailed": []
    },
    "economic_environment": {
        "title": "Economic Environment", 
        "color": "red",
        "basic": ["ECONOMIC_STRESS_LEVEL"],
        "detailed": ["DOWNTURN_JOIN_MULT", "DOWNTURN_CHURN_MULT"]
    },
    "financing": {
        "title": "SBA Loan Financing", 
        "color": "red",
        "basic": ["LOAN_504_ANNUAL_RATE", "LOAN_7A_ANNUAL_RATE"],
        "detailed": ["LOAN_504_TERM_YEARS", "LOAN_7A_TERM_YEARS", "IO_MONTHS_504", "IO_MONTHS_7A", 
                    "LOAN_CONTINGENCY_PCT", "RUNWAY_MONTHS", "EXTRA_BUFFER", "RESERVE_FLOOR"]
    },
    "grants": {
        "title": "Grants & External Funding", 
        "color": "red",
        "basic": ["grant_amount", "grant_month"],
        "detailed": []
    }
}

# Keep existing helper functions but update parameter references
def consolidate_build_overrides(env, strat):
    """
    Fixed to match the function call: consolidate_build_overrides(env, strat)
    Maps consolidated UI parameters to simulator expected parameters
    """
    
    # Combine env and strat into scenario_params (like your original build_overrides did)
    scenario_params = {}
    if env:
        scenario_params.update(env)
    if strat:
        scenario_params.update(strat)
    
    print("=== DEBUG: consolidate_build_overrides INPUT ===")
    print(f"env keys: {list(env.keys()) if env else 'None'}")
    print(f"strat keys: {list(strat.keys()) if strat else 'None'}")
    print(f"combined scenario_params keys: {list(scenario_params.keys())}")
    for key, value in scenario_params.items():
        print(f"  {key}: {value} (type: {type(value)})")
    
    # Create the mapping from consolidated params to simulator params
    overrides = {}
    
    # Map consolidated parameters to what simulator expects
    param_mapping = {
        # Consolidated -> Simulator expected
        'MONTHLY_RENT': 'RENT',
        'OWNER_COMPENSATION': 'OWNER_DRAW',
        # Add ALL your consolidated parameter mappings here
        # Look at your CONSOLIDATED_PARAM_SPECS vs original PARAM_SPECS
    }
    
    print(f"\n=== DEBUG: Parameter mapping rules ===")
    for consolidated, simulator_expected in param_mapping.items():
        print(f"  {consolidated} -> {simulator_expected}")
    
    # Apply mappings
    for consolidated_name, simulator_name in param_mapping.items():
        if consolidated_name in scenario_params:
            overrides[simulator_name] = scenario_params[consolidated_name]
            print(f"  MAPPED: {consolidated_name} ({scenario_params[consolidated_name]}) -> {simulator_name}")
    
    # Pass through any parameters not in mapping (unchanged parameters)
    for param_name, value in scenario_params.items():
        if param_name not in param_mapping:
            overrides[param_name] = value
            print(f"  PASSTHROUGH: {param_name} -> {value}")
    
    print(f"\n=== DEBUG: consolidate_build_overrides OUTPUT ===")
    print(f"overrides keys: {list(overrides.keys())}")
    for key, value in overrides.items():
        print(f"  {key}: {value}")
    
    return overrides


# Debug wrapper for your simulation call
def debug_simulation_call(scenario_params):
    """
    Wrap your simulation call with debugging
    """
    print("\n" + "="*50)
    print("DEBUGGING SIMULATION CALL")
    print("="*50)
    
    # Step 1: Debug parameter consolidation
    overrides = consolidate_build_overrides(scenario_params)
    
    # Step 2: Debug what gets passed to simulator
    print(f"\n=== CALLING SIMULATOR WITH ===")
    print(f"overrides: {overrides}")
    
    try:
        # Replace this with your actual simulator call
        # results = your_simulator_function(overrides)
        
        # For debugging, let's see what modular_simulator expects
        print(f"\n=== CHECKING SIMULATOR EXPECTATIONS ===")
        # You'll need to check what parameters modular_simulator.py actually expects
        # Look for parameter names in modular_simulator.py
        
        # Simulate the call (replace with actual)
        # results = run_simulation_with_overrides(overrides)
        
        print(f"Simulation would be called with these parameters:")
        for key, value in overrides.items():
            print(f"  {key}: {value}")
            
        # Check if results would be empty
        # if results is None or len(results) == 0:
        #     print("WARNING: Simulator returned empty results!")
        #     return None
            
        return overrides  # Return for inspection
        
    except Exception as e:
        print(f"ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


# Function to check what modular_simulator.py expects
def check_simulator_expectations():
    """
    Debug function to check what parameters the simulator actually expects
    """
    print("\n=== CHECKING SIMULATOR PARAMETER EXPECTATIONS ===")
    
    # You need to inspect modular_simulator.py to see what parameters it expects
    # Look for:
    # 1. Function signatures
    # 2. Parameter access (like params['RENT'] vs params['MONTHLY_RENT'])
    # 3. Default parameter definitions
    
    expected_params = [
        # Add the actual parameter names that modular_simulator.py expects
        # These are likely the OLD parameter names before consolidation
        'RENT',  # not MONTHLY_RENT
        'OWNER_DRAW',  # not OWNER_COMPENSATION
        # Add others...
    ]
    
    print("Simulator likely expects these parameter names:")
    for param in expected_params:
        print(f"  - {param}")
    
    return expected_params


# Quick fix function - maps ALL consolidated params to simulator expectations
def create_complete_mapping():
    """
    Create complete mapping from consolidated parameters to simulator parameters
    """
    # You need to fill this based on your consolidation
    complete_mapping = {
        # Consolidated UI name -> Simulator expected name
        'MONTHLY_RENT': 'RENT',
        'OWNER_COMPENSATION': 'OWNER_DRAW',
        # Add ALL your consolidated parameters here
        # Look at CONSOLIDATED_PARAM_SPECS and match to original PARAM_SPECS
    }
    
    return complete_mapping


# Test function to validate the mapping
def test_parameter_mapping(sample_scenario_params):
    """
    Test the parameter mapping with sample data
    """
    print("\n=== TESTING PARAMETER MAPPING ===")
    
    # Test with sample data
    test_params = sample_scenario_params or {
        'MONTHLY_RENT': 5000,
        'OWNER_COMPENSATION': 3000,
        # Add other test parameters
    }
    
    print(f"Testing with: {test_params}")
    
    # Test the mapping
    result = debug_simulation_call(test_params)
    
    # Validate result has expected parameters
    expected = check_simulator_expectations()
    
    if result:
        missing = [p for p in expected if p not in result]
        if missing:
            print(f"WARNING: Missing expected parameters: {missing}")
        else:
            print("✓ All expected parameters present")
    
    return result

def render_consolidated_parameter_group(group_name, group_config, params_state, prefix=""):
    """Render consolidated parameter group with progressive disclosure"""
    color_indicators = {"green": "🟢", "amber": "🟡", "red": "🔴"}
    color_descriptions = {
        "green": "Most likely to vary between studios",
        "amber": "May need adjustment for your situation", 
        "red": "Set once during planning"
    }
    
    st.markdown(f"**{group_config['title']}**")
    color = group_config.get('color', 'amber')
    st.caption(f"{color_indicators[color]} {color_descriptions[color]}")
    
    # Always show basic parameters
    for param_name in group_config['basic']:
        if param_name in CONSOLIDATED_PARAM_SPECS:
            spec = CONSOLIDATED_PARAM_SPECS[param_name]
            params_state[param_name] = render_consolidated_parameter(param_name, spec, params_state.get(param_name), prefix)
    
    # Detailed parameters in expander if they exist
    if group_config.get('detailed'):
        with st.expander("🔧 Advanced Settings", expanded=False):
            for param_name in group_config['detailed']:
                if param_name in CONSOLIDATED_PARAM_SPECS:
                    spec = CONSOLIDATED_PARAM_SPECS[param_name]
                    params_state[param_name] = render_consolidated_parameter(param_name, spec, params_state.get(param_name), prefix)
    
    return params_state

def render_consolidated_parameter(param_name, spec, current_value, prefix=""):
    """Render individual consolidated parameter with appropriate widget"""
    
    param_type = spec['type']
    label = spec['label']
    help_text = build_consolidated_help_text(spec)
    key = f"{prefix}_{param_name}" if prefix else param_name
    
    # Set default if needed
    if current_value is None:
        current_value = spec.get('default', get_consolidated_default(spec))
    
    # Render widget based on type
    if param_type == 'bool':
        return st.checkbox(label, value=current_value, key=key, help=help_text)
    
    elif param_type == 'int':
        value = st.slider(
            label, 
            min_value=spec['min'], 
            max_value=spec['max'], 
            value=current_value,
            step=spec['step'],
            key=key,
            help=help_text
        )
        show_range_hint(value, spec)
        return value
    
    elif param_type == 'float':
        value = st.slider(
            label,
            min_value=spec['min'],
            max_value=spec['max'], 
            value=current_value,
            step=spec['step'],
            key=key,
            help=help_text
        )
        show_range_hint(value, spec)
        return value
    
    elif param_type == 'select':
        options = spec['options']
        if isinstance(options[0], tuple):
            try:
                current_index = next(i for i, opt in enumerate(options) 
                                   if (isinstance(current_value, tuple) and opt[1] == current_value[1]) 
                                   or opt[1] == current_value)
            except (StopIteration, TypeError):
                current_index = 0
            
            return st.selectbox(
                label,
                options=options,
                index=current_index,
                format_func=lambda x: x[0] if isinstance(x, tuple) else str(x),
                key=key,
                help=help_text
            )
        else:
            current_index = options.index(current_value) if current_value in options else 0
            return st.selectbox(label, options=options, index=current_index, key=key, help=help_text)
    
    elif param_type == 'market_inflow':
        # Keep your existing market inflow rendering logic
        base = f"{key}"
        cur = _normalize_market_inflow(current_value if isinstance(current_value, dict) else {})
        c_def = st.session_state.get(f"{base}_c", cur["community_studio"])
        h_def = st.session_state.get(f"{base}_h", cur["home_studio"])
        n_def = st.session_state.get(f"{base}_n", cur["no_access"])
    
        c = st.slider("Community studio inflow", 0, 50, int(c_def), key=f"{base}_c", help=help_text)
        h = st.slider("Home studio inflow",      0, 50, int(h_def), key=f"{base}_h", help=help_text)
        n = st.slider("No access inflow",        0, 50, int(n_def), key=f"{base}_n", help=help_text)
    
        result = {"community_studio": c, "home_studio": h, "no_access": n}
        st.session_state[base] = result
        return result
    
    return current_value

def build_consolidated_help_text(spec):
    """Build help text from consolidated spec"""
    parts = []
    
    if 'desc' in spec:
        parts.append(spec['desc'])
    
    if 'rec' in spec and isinstance(spec['rec'], (list, tuple)) and len(spec['rec']) == 2:
        parts.append(f"Typical range: {spec['rec'][0]} - {spec['rec'][1]}")
    
    return " | ".join(parts)

def show_range_hint(value, spec):
    """Show hint if value is outside recommended range"""
    try:
        if not st.session_state.get("_show_hints", True):
            return
        rec = spec.get("rec")
        if isinstance(rec, (list, tuple)) and len(rec) == 2:
            lo, hi = float(rec[0]), float(rec[1])
            if value < lo or value > hi:
                st.caption(f"⚠️ Outside typical range ({lo}-{hi}). Consider if this fits your situation.")
    except Exception:
        pass

def get_consolidated_default(spec):
    """Get default value for consolidated parameter"""
    if 'default' in spec:
        return spec['default']
    elif spec['type'] == 'bool':
        return False
    elif spec['type'] in ['int', 'float']:
        return spec['min']
    elif spec['type'] == 'select':
        return spec['options'][0]
    return None

def _normalize_market_inflow(d: dict) -> dict:
    """Normalize market inflow data"""
    pools = {
        "community_studio": d.get("community_studio", 0),
        "home_studio":      d.get("home_studio", 0),
        "no_access":        d.get("no_access", 0),
    }
    out = {}
    for k, v in pools.items():
        try:
            out[k] = max(0, int(v))
        except Exception:
            out[k] = 0
    return out

# Keep all your existing helper functions
def compute_kpis_from_cell(df_cell: pd.DataFrame) -> dict:
    """Compute lender-style KPIs from a single cell's simulation dataframe"""
    out = {}
    if df_cell.empty:
        return out

    # Resolve columns
    month_col = "month" if "month" in df_cell.columns else ("Month" if "Month" in df_cell.columns else "t")
    if month_col not in df_cell.columns:
        return out

    cash_col = pick_col(df_cell, ["cash_balance", "cash", "ending_cash"])
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

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# Keep your existing cache and other helper functions
def _preflight_validate(cfg: dict) -> bool:
    """Return True if config looks sane"""
    errs = []
    if "USAGE_SHARE" in cfg and not isinstance(cfg["USAGE_SHARE"], (dict, list, tuple)):
        errs.append("USAGE_SHARE must be a dict/list")
    if "STATIONS" in cfg and not isinstance(cfg["STATIONS"], (dict, list, int)):
        errs.append("STATIONS must be dict/list/int")
    if errs:
        st.error("Invalid inputs:\n- " + "\n- ".join(errs))
        return False
    return True

# Keep your existing caching decorators and simulation functions
@st.cache_data(show_spinner=False)
def get_defaults_cached():
    from modular_simulator import get_default_cfg
    return get_default_cfg()

def _normalize_capex_items(df):
    """Convert the data_editor DataFrame into a clean list[dict]"""
    items = []
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return items
    for _, r in df.iterrows():
        try:
            label = str(r.get("label", "")).strip()
            unit  = float(r.get("unit_cost", 0) or 0)
            cnt   = int(r.get("count", 1) or 1)
            mth   = r.get("month", None)
            thr   = r.get("member_threshold", None)
            enabled = bool(r.get("enabled", True))
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

# Keep your existing figure capture and caching
class FigureCapture:
    def __init__(self, title_suffix: str = ""):
        self.title_suffix = title_suffix
        self._orig_show = None
        self.images: List[Tuple[str, bytes]] = []
        self.manifest = []

    def __enter__(self):
        matplotlib.use("Agg", force=True)
        self._orig_show = plt.show
        counter = {"i": 0}

        def _title_for(fig):
            parts = []
            if fig._suptitle:
                txt = fig._suptitle.get_text()
                if txt:
                    parts.append(txt)
            for ax in fig.axes:
                t = getattr(ax, "get_title", lambda: "")()
                if t:
                    parts.append(t)
            return " | ".join(parts).strip()

        def _ensure_suffix(fig):
            if not self.title_suffix:
                return
            has_any_title = any(ax.get_title() for ax in fig.get_axes())
            if not has_any_title:
                fig.suptitle(self.title_suffix)

        def _show(*args, **kwargs):
            counter["i"] += 1
            fig = plt.gcf()
        
            _ensure_suffix(fig)
        
            has_suptitle  = bool(fig._suptitle and fig._suptitle.get_text())
            has_ax_titles = any(ax.get_title() for ax in fig.get_axes())
        
            if has_suptitle and has_ax_titles:
                fig._suptitle.set_y(0.98)
                try:
                    fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize() - 2, 10))
                except Exception:
                    pass
                fig.tight_layout(rect=[0, 0, 1, 0.94])
            else:
                fig.tight_layout()
        
            buf = io.BytesIO()
            fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
            buf.seek(0)
            fname = f"fig_{counter['i']:02d}.png"
            self.images.append((fname, buf.read()))
            self.manifest.append({"file": fname, "title": _title_for(fig)})
            plt.close(fig)

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_show:
            plt.show = self._orig_show

@st.cache_data(show_spinner=False)
def run_cell_cached(env: dict, strat: dict, seed: int, cache_key: Optional[str] = None):
    """Your existing cached simulation runner"""
    if cache_key is None:
        cache_key = f"v6|{json.dumps(env, sort_keys=True)}|{json.dumps(strat, sort_keys=True)}|{seed}"

    ov = consolidate_build_overrides(env, strat)  # Use consolidated function
    st.write("DEBUG - Parameters generated:")
    st.write({k: v for k, v in ov.items() if not k.startswith('CAPEX')})  # Show key params
    ov["RANDOM_SEED"] = seed

    title_suffix = f"{env['name']} | {strat['name']}"
    with FigureCapture(title_suffix) as cap:
        try:
            res = run_original_once("modular_simulator.py", ov)
            st.write("DEBUG - Simulation result type:", type(res))
            if isinstance(res, tuple):
                st.write("DEBUG - DataFrame shape:", res[0].shape if res[0] is not None else "None")
                st.write("DEBUG - DataFrame columns:", list(res[0].columns) if res[0] is not None else "None")
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)
            return pd.DataFrame(), None, [], []

    df_cell, eff = (res if isinstance(res, tuple) else (res, None))

    df_cell = df_cell.copy()
    df_cell["environment"] = env["name"]
    df_cell["strategy"]    = strat["name"]
    if "simulation_id" not in df_cell.columns:
        df_cell["simulation_id"] = 0
    return df_cell, eff, cap.images, cap.manifest

# Keep your existing summarize_cell and other analysis functions
def summarize_cell(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """Your existing cell summary function"""
    if df.empty:
        return {}, pd.DataFrame()
    
    env_col = "environment"
    strat_col = "strategy"
    sim_col = "simulation_id"
    month_col = "month" if "month" in df.columns else ("Month" if "Month" in df.columns else "t")
    
    if month_col not in df.columns:
        return {}, pd.DataFrame()

    cash_col = pick_col(df, ["cash_balance","cash","ending_cash"])
    cf_col   = pick_col(df, ["cfads","operating_cash_flow","op_cf","net_cash_flow","cash_flow"])

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
            "t_breakeven":  _first_sustained_ge_zero(g, k=breakeven_k),
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

# Main UI
st.set_page_config(page_title="GCWS Simulator - Enhanced", layout="wide")
st.title("Ginkgo Clayworks — Scenario Explorer (Enhanced)")

# Your existing scenarios and strategies
SCENARIOS = [
    {
        "name": "Baseline",
        "ECONOMIC_STRESS_LEVEL": ("Moderate", 0.08),
        "DOWNTURN_JOIN_MULT": 1.00,
        "DOWNTURN_CHURN_MULT": 1.00,
        "MARKET_POOLS_INFLOW": {"community_studio": 4, "home_studio": 2, "no_access": 3},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.03,
        "LEAD_TO_JOIN_RATE": 0.20,
        "MAX_ONBOARD_PER_MONTH": 10,
        "STUDIO_CAPACITY": 92,
        "EXPANSION_THRESHOLD": 20,
    },
    {
        "name": "Recession", 
        "ECONOMIC_STRESS_LEVEL": ("Stressed", 0.18),
        "DOWNTURN_JOIN_MULT": 0.65,
        "DOWNTURN_CHURN_MULT": 1.50,
        "MARKET_POOLS_INFLOW": {"community_studio": 2, "home_studio": 1, "no_access": 1},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.02,
        "LEAD_TO_JOIN_RATE": 0.15,
        "MAX_ONBOARD_PER_MONTH": 8,
        "STUDIO_CAPACITY": 86,
        "EXPANSION_THRESHOLD": 25,
    },
    {
        "name": "SlowRecovery_Grant25k_M4",
        "ECONOMIC_STRESS_LEVEL": ("Uncertain", 0.10),
        "DOWNTURN_JOIN_MULT": 0.85,
        "DOWNTURN_CHURN_MULT": 1.20,
        "MARKET_POOLS_INFLOW": {"community_studio": 3, "home_studio": 1, "no_access": 2},
        "grant_amount": 25000, "grant_month": 4,
        "WOM_RATE": 0.025,
        "LEAD_TO_JOIN_RATE": 0.18,
        "MAX_ONBOARD_PER_MONTH": 9,
        "STUDIO_CAPACITY": 94,
        "EXPANSION_THRESHOLD": 22,
    },
    {
        "name": "Boom",
        "ECONOMIC_STRESS_LEVEL": ("Normal", 0.02),
        "DOWNTURN_JOIN_MULT": 1.20,
        "DOWNTURN_CHURN_MULT": 0.85,
        "MARKET_POOLS_INFLOW": {"community_studio": 6, "home_studio": 3, "no_access": 4},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.04,
        "LEAD_TO_JOIN_RATE": 0.25,
        "MAX_ONBOARD_PER_MONTH": 12,
        "STUDIO_CAPACITY": 100,
        "EXPANSION_THRESHOLD": 18,
    },
]

STRATEGIES = [
    {"name":"Enhanced_A", "MONTHLY_RENT":4000, "OWNER_COMPENSATION":2000, "MEMBERSHIP_PRICE": 185,
     "CLASS_SCHEDULE_MODE": "semester", "CLASSES_PER_PERIOD": 2},
    {"name":"Enhanced_B", "MONTHLY_RENT":4000, "OWNER_COMPENSATION":2000, "MEMBERSHIP_PRICE": 185,
     "CLASS_SCHEDULE_MODE": "semester", "CLASSES_PER_PERIOD": 2},
]

# Sidebar with consolidated controls
with st.sidebar:
    with st.expander("About this model", expanded=False):
        st.markdown("Enhanced version with consolidated parameters and improved visualization.")
    
    st.header("Configuration")
    st.caption("Hover over any label for explanation. Colors indicate how likely parameters are to vary between studios.")
    st.session_state["_show_hints"] = st.toggle("Show range hints", value=True)

    scen_names  = [s["name"] for s in SCENARIOS]
    strat_names = [s["name"] for s in STRATEGIES]

    scen_sel  = st.selectbox("Scenario preset", scen_names, index=0)
    strat_sel = st.selectbox("Strategy preset", strat_names, index=0)

    # Loan controls (keep your existing loan UI)
    st.subheader("Loans")
    colA, colB = st.columns(2)
    with colA:
        capex_mode = st.radio("CapEx Loan (504) Mode", ["upfront","staged"], index=0, horizontal=True)
        if capex_mode == "upfront":
            loan_504 = st.number_input("Upfront CapEx Loan (504)", min_value=0, step=1000, value=0)
        else:
            capex_draw_pct = st.slider("CapEx: Staged Draw %", 0.0, 1.0, 1.0, 0.05)
            capex_min_tr   = st.number_input("CapEx: Min Tranche ($)", min_value=0, step=500, value=0)
            capex_max_tr   = st.number_input("CapEx: Max Tranche ($)", min_value=0, step=500, value=0)
            
    with colB:
        opex_mode = st.radio("OpEx Loan (7a) Mode", ["upfront","staged"], index=0, horizontal=True)
        if opex_mode == "upfront":
            loan_7a  = st.number_input("Upfront OpEx Loan (7a)",  min_value=0, step=1000, value=0)
        else:
            opex_facility = st.number_input("OpEx: Facility Limit ($)", min_value=0, step=1000, value=0)
            opex_min_tr   = st.number_input("OpEx: Min Draw ($)",     min_value=0, step=500,  value=0)
            opex_max_tr   = st.number_input("OpEx: Max Draw ($)",     min_value=0, step=500,  value=0)
            reserve_floor = st.number_input("OpEx: Cash Floor ($)",    min_value=0, step=500,  value=0)

    # Store loan settings
    st.session_state["capex_mode"] = capex_mode
    st.session_state["opex_mode"]  = opex_mode
    if capex_mode == "upfront":
        st.session_state["loan_504"] = float(loan_504)
    else:
        st.session_state["capex_draw_pct"] = float(capex_draw_pct)
        st.session_state["capex_min_tr"]   = float(capex_min_tr)
        st.session_state["capex_max_tr"]   = float(capex_max_tr)
    if opex_mode == "upfront":
        st.session_state["loan_7a"] = float(loan_7a)
    else:
        st.session_state["opex_facility"]  = float(opex_facility)
        st.session_state["opex_min_tr"]    = float(opex_min_tr)
        st.session_state["opex_max_tr"]    = float(opex_max_tr)
        st.session_state["reserve_floor"]  = float(reserve_floor)
   
    # Simulation settings
    with st.expander("Simulation settings", expanded=False):
        sim_count = st.slider("Simulations per run", min_value=5, max_value=300, step=5, value=100)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, step=1, value=42)
    st.session_state["N_SIMULATIONS"] = int(sim_count)
    st.session_state["RANDOM_SEED"]   = int(seed)

    # Get selected presets
    env  = next(s for s in SCENARIOS  if s["name"] == scen_sel)
    strat = next(s for s in STRATEGIES if s["name"] == strat_sel)
    strat["N_SIMULATIONS"] = int(sim_count)
    
    # Render consolidated parameter groups
    for group_name, group_config in CONSOLIDATED_GROUPS.items():
        with st.expander(group_config['title'], expanded=(group_name == 'business_core')):
            if group_name in ['business_core', 'market_response']:
                env = render_consolidated_parameter_group(group_name, group_config, env, "env")
            else:
                strat = render_consolidated_parameter_group(group_name, group_config, strat, "strat")

    # Equipment section (keep your existing data_editor)
    with st.expander("Equipment", expanded=True):
        st.markdown("**Staged purchases (all equipment)**")
        
        capex_existing = strat.get("CAPEX_ITEMS", [])
        capex_df_default = pd.DataFrame(capex_existing) if capex_existing else pd.DataFrame([
            {"enabled": True,  "label": "Kiln #1 Skutt 1227", "count": 1,  "unit_cost": 7000, "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Wheels",       "count": 12,  "unit_cost": 3000,  "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Wire racks",   "count": 5,  "unit_cost": 150,  "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Clay traps",   "count": 1,  "unit_cost": 160,  "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Kiln #2 Skutt 1427", "count": 1,  "unit_cost": 10000, "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Wire racks",   "count": 7,  "unit_cost": 150,  "month": 0,   "member_threshold": None, "finance_504": True},
            {"enabled": False, "label": "Wheels",       "count": 10, "unit_cost": 800,  "month": 6,   "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Slab roller",  "count": 1,  "unit_cost": 1800, "month": None,"member_threshold": 50, "finance_504": True},
        ])
        
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
        
        strat["CAPEX_ITEMS"] = _normalize_capex_items(capex_df)

# Main content - single run tab
if not _preflight_validate(strat):
    st.stop()

st.caption("Enhanced app with consolidated parameters and improved visualization")
run_btn = st.button("Run simulation")

if run_btn:
    try:
        run_cell_cached.clear()
        st.cache_data.clear()
    except Exception:
        pass
        
    with st.spinner("Running simulator…"):
        cache_key = f"v7|{json.dumps(env, sort_keys=True)}|{json.dumps(strat, sort_keys=True)}|{seed}"
        df_cell, eff, images, manifest = run_cell_cached(env, strat, seed, cache_key)
        st.session_state["df_result"] = df_cell
        
    st.subheader(f"Results — {env['name']} | {strat['name']}")

    # Display KPIs and results using your existing functions
    kpi_cell = compute_kpis_from_cell(df_cell)
    row_dict, _tim = summarize_cell(df_cell)
    
    # Show metrics
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
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for fname, data in images:
            zf.writestr(fname, data)
    st.download_button("Download plots (zip)", data=buf.getvalue(),
                        file_name=f"{env['name']}__{strat['name']}_plots.zip")

    st.markdown("#### Raw results")
    st.dataframe(df_cell.head(250))