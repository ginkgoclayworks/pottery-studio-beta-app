#!/usr/bin/env python3
"""
Reusable UI components for the Streamlit interface.
Handles parameter rendering, progressive disclosure, and user interactions.
"""

import streamlit as st
from config.parameters import CONSOLIDATED_PARAM_SPECS, CONSOLIDATED_GROUPS
from simulation.validation import normalize_market_inflow


def render_parameter_group(group_name: str, group_config: dict, params_state: dict, prefix: str = ""):
    """
    Render consolidated parameter group with progressive disclosure.
    
    Args:
        group_name: Name of the parameter group
        group_config: Configuration for the group
        params_state: Current parameter state
        prefix: Prefix for widget keys
    
    Returns:
        Updated parameter state
    """
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
            params_state[param_name] = render_parameter(param_name, spec, params_state.get(param_name), prefix)
    
    # Detailed parameters in expander if they exist
    if group_config.get('detailed'):
        with st.expander("🔧 Advanced Settings", expanded=False):
            for param_name in group_config['detailed']:
                if param_name in CONSOLIDATED_PARAM_SPECS:
                    spec = CONSOLIDATED_PARAM_SPECS[param_name]
                    params_state[param_name] = render_parameter(param_name, spec, params_state.get(param_name), prefix)
    
    return params_state


def render_parameter(param_name: str, spec: dict, current_value, prefix: str = ""):
    """
    Render individual parameter with appropriate widget.
    
    Args:
        param_name: Name of the parameter
        spec: Parameter specification
        current_value: Current parameter value
        prefix: Prefix for widget keys
    
    Returns:
        Updated parameter value
    """
    param_type = spec['type']
    label = spec['label']
    help_text = build_help_text(spec)
    key = f"{prefix}_{param_name}" if prefix else param_name
    
    # Set default if needed
    if current_value is None:
        current_value = spec.get('default', get_default_value(spec))
    
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
        return render_market_inflow(key, current_value, help_text)
    
    return current_value


def render_market_inflow(key: str, current_value, help_text: str):
    """
    Render market inflow parameter with multiple sliders.
    
    Args:
        key: Widget key
        current_value: Current inflow values
        help_text: Help text for the parameter
    
    Returns:
        Updated market inflow dictionary
    """
    cur = normalize_market_inflow(current_value if isinstance(current_value, dict) else {})
    c_def = st.session_state.get(f"{key}_c", cur["community_studio"])
    h_def = st.session_state.get(f"{key}_h", cur["home_studio"])
    n_def = st.session_state.get(f"{key}_n", cur["no_access"])

    c = st.slider("Community studio inflow", 0, 50, int(c_def), key=f"{key}_c", help=help_text)
    h = st.slider("Home studio inflow",      0, 50, int(h_def), key=f"{key}_h", help=help_text)
    n = st.slider("No access inflow",        0, 50, int(n_def), key=f"{key}_n", help=help_text)

    result = {"community_studio": c, "home_studio": h, "no_access": n}
    st.session_state[key] = result
    return result


def render_loan_controls():
    """
    Render loan configuration controls in sidebar.
    
    Returns:
        Dictionary of loan settings
    """
    st.subheader("Loans")
    colA, colB = st.columns(2)
    
    with colA:
        capex_mode = st.radio("CapEx Loan (504) Mode", ["upfront","staged"], index=0, horizontal=True)
        if capex_mode == "upfront":
            loan_504 = st.number_input("Upfront CapEx Loan (504)", min_value=0, step=1000, value=0)
        else:
            capex_draw_pct = st.slider("CapEx: Staged Draw %", 0.0, 1.0, 1.0, 0.05)
            capex_min_tr = st.number_input("CapEx: Min Tranche ($)", min_value=0, step=500, value=0)
            capex_max_tr = st.number_input("CapEx: Max Tranche ($)", min_value=0, step=500, value=0)
            
    with colB:
        opex_mode = st.radio("OpEx Loan (7a) Mode", ["upfront","staged"], index=0, horizontal=True)
        if opex_mode == "upfront":
            loan_7a = st.number_input("Upfront OpEx Loan (7a)",  min_value=0, step=1000, value=0)
        else:
            opex_facility = st.number_input("OpEx: Facility Limit ($)", min_value=0, step=1000, value=0)
            opex_min_tr = st.number_input("OpEx: Min Draw ($)", min_value=0, step=500,  value=0)
            opex_max_tr = st.number_input("OpEx: Max Draw ($)", min_value=0, step=500,  value=0)
            reserve_floor = st.number_input("OpEx: Cash Floor ($)", min_value=0, step=500,  value=0)

    # Build loan settings dictionary
    loan_settings = {
        "capex_mode": capex_mode,
        "opex_mode": opex_mode,
    }
    
    if capex_mode == "upfront":
        loan_settings["loan_504"] = float(loan_504)
    else:
        loan_settings.update({
            "capex_draw_pct": float(capex_draw_pct),
            "capex_min_tr": float(capex_min_tr),
            "capex_max_tr": float(capex_max_tr),
        })
    
    if opex_mode == "upfront":
        loan_settings["loan_7a"] = float(loan_7a)
    else:
        loan_settings.update({
            "opex_facility": float(opex_facility),
            "opex_min_tr": float(opex_min_tr),
            "opex_max_tr": float(opex_max_tr),
            "reserve_floor": float(reserve_floor),
        })
    
    return loan_settings


def render_simulation_settings():
    """
    Render simulation configuration controls.
    
    Returns:
        Dictionary of simulation settings
    """
    with st.expander("Simulation settings", expanded=False):
        sim_count = st.slider("Simulations per run", min_value=5, max_value=300, step=5, value=100)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, step=1, value=42)
    
    return {
        "N_SIMULATIONS": int(sim_count),
        "RANDOM_SEED": int(seed)
    }


def build_help_text(spec: dict) -> str:
    """
    Build help text from parameter specification.
    
    Args:
        spec: Parameter specification dictionary
    
    Returns:
        Formatted help text
    """
    parts = []
    
    if 'desc' in spec:
        parts.append(spec['desc'])
    
    if 'rec' in spec and isinstance(spec['rec'], (list, tuple)) and len(spec['rec']) == 2:
        parts.append(f"Typical range: {spec['rec'][0]} - {spec['rec'][1]}")
    
    return " | ".join(parts)


def show_range_hint(value, spec: dict):
    """
    Show hint if value is outside recommended range.
    
    Args:
        value: Current parameter value
        spec: Parameter specification
    """
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


def get_default_value(spec: dict):
    """
    Get default value for parameter specification.
    
    Args:
        spec: Parameter specification
    
    Returns:
        Default value for the parameter
    """
    if 'default' in spec:
        return spec['default']
    elif spec['type'] == 'bool':
        return False
    elif spec['type'] in ['int', 'float']:
        return spec['min']
    elif spec['type'] == 'select':
        return spec['options'][0]
    return None
