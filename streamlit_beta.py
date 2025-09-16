#!/usr/bin/env python3
"""
streamlit_beta.py

User-friendly Streamlit app for pottery studio business modeling.
Designed for studio owners who need financial projections for SBA loans
or business planning but aren't familiar with complex financial modeling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config_manager import StudioConfig, create_preset_config, SCENARIO_PRESETS
from core_simulator import run_simulation, run_scenario_comparison, create_quick_comparison_table

# Page configuration
st.set_page_config(
    page_title="Pottery Studio Business Simulator",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

def show_help_text(text, key=None):
    """Display expandable help text"""
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.write(text)

def render_business_fundamentals(config):
    """Render the business fundamentals section"""
    st.subheader("Business Fundamentals")
    st.write("Core operational parameters that define your studio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.business.monthly_rent = st.number_input(
            "Monthly Rent ($)",
            min_value=500.0,
            max_value=15000.0,
            value=config.business.monthly_rent,
            step=100.0,
            help="Your studio's monthly lease payment"
        )
        
        config.business.membership_price = st.number_input(
            "Monthly Membership Price ($)",
            min_value=25.0,
            max_value=200.0,
            value=config.business.membership_price,
            step=5.0,
            help="What you charge members per month for studio access"
        )
    
    with col2:
        config.business.max_members = st.number_input(
            "Studio Capacity (Max Members)",
            min_value=30,
            max_value=300,
            value=config.business.max_members,
            step=10,
            help="Maximum number of members your studio can accommodate"
        )
        
        config.business.owner_monthly_compensation = st.number_input(
            "Owner Monthly Draw ($)",
            min_value=0.0,
            max_value=8000.0,
            value=config.business.owner_monthly_compensation,
            step=100.0,
            help="Monthly income you plan to take from the business"
        )

def render_financing_config(config):
    """Render the financing configuration section"""
    st.subheader("SBA Loan Financing")
    st.write("Configure your SBA loan terms and working capital needs")
    
    show_help_text("""
    **SBA 504 Loan**: Typically used for real estate and major equipment. Lower rates, longer terms.
    **SBA 7(a) Loan**: More flexible, can cover working capital, equipment, leasehold improvements.
    **Interest-Only Period**: Initial months where you only pay interest, not principal. Helps with cash flow but increases total cost.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**SBA 504 Loan**")
        config.financing.loan_504_amount = st.number_input(
            "504 Loan Amount ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=config.financing.loan_504_amount,
            step=10000.0
        )
        
        config.financing.loan_504_rate = st.slider(
            "504 Interest Rate (%)",
            min_value=4.0,
            max_value=12.0,
            value=config.financing.loan_504_rate * 100,
            step=0.25
        ) / 100
        
        config.financing.loan_504_term_years = st.selectbox(
            "504 Term (Years)",
            options=[10, 15, 20, 25],
            index=2,  # Default to 20 years
        )
        
        config.financing.loan_504_io_months = st.slider(
            "504 Interest-Only Months",
            min_value=0,
            max_value=24,
            value=config.financing.loan_504_io_months,
            step=1
        )
    
    with col2:
        st.write("**SBA 7(a) Loan**")
        config.financing.loan_7a_amount = st.number_input(
            "7(a) Loan Amount ($)",
            min_value=0.0,
            max_value=500000.0,
            value=config.financing.loan_7a_amount,
            step=5000.0
        )
        
        config.financing.loan_7a_rate = st.slider(
            "7(a) Interest Rate (%)",
            min_value=5.0,
            max_value=15.0,
            value=config.financing.loan_7a_rate * 100,
            step=0.25
        ) / 100
        
        config.financing.loan_7a_term_years = st.selectbox(
            "7(a) Term (Years)",
            options=[5, 7, 10, 15],
            index=2,  # Default to 10 years
        )
        
        config.financing.loan_7a_io_months = st.slider(
            "7(a) Interest-Only Months",
            min_value=0,
            max_value=18,
            value=config.financing.loan_7a_io_months,
            step=1
        )
    
    st.write("**Working Capital & Reserves**")
    col3, col4 = st.columns(2)
    
    with col3:
        config.financing.working_capital_target = st.number_input(
            "Working Capital Target ($)",
            min_value=5000.0,
            max_value=200000.0,
            value=config.financing.working_capital_target,
            step=5000.0,
            help="Emergency cash buffer for unexpected expenses"
        )
    
    with col4:
        config.financing.runway_months = st.slider(
            "Cash Runway Target (Months)",
            min_value=6,
            max_value=36,
            value=config.financing.runway_months,
            step=1,
            help="How many months of expenses you want to be able to cover"
        )

def render_equipment_config(config):
    """Render the equipment configuration section"""
    st.subheader("Starting Equipment")
    st.write("Equipment and infrastructure at studio opening")
    
    show_help_text("""
    **Pottery Wheels**: Plan 1 wheel per 8-12 members. More wheels = higher capacity but higher cost.
    **Drying Racks**: Plan 1 rack per 6-8 members for adequate drying space.
    **Slab Roller**: Enables hand-building classes, adds ~$3,000 to startup costs.
    **Pug Mill**: For clay recycling, saves ~$200/month in clay costs but costs ~$8,000 upfront.
    **Clay Traps**: Minimum 2 for proper studio drainage, 1 per sink area.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        config.equipment.pottery_wheels = st.number_input(
            "Pottery Wheels",
            min_value=1,
            max_value=20,
            value=config.equipment.pottery_wheels,
            step=1
        )
        
        config.equipment.drying_racks = st.number_input(
            "Drying Racks",
            min_value=1,
            max_value=25,
            value=config.equipment.drying_racks,
            step=1
        )
    
    with col2:
        config.equipment.has_slab_roller = st.checkbox(
            "Include Slab Roller",
            value=config.equipment.has_slab_roller,
            help="Enables hand-building classes"
        )
        
        config.equipment.has_pug_mill = st.checkbox(
            "Include Pug Mill",
            value=config.equipment.has_pug_mill,
            help="For clay recycling - saves money long-term"
        )
    
    with col3:
        config.equipment.clay_traps = st.number_input(
            "Clay Traps",
            min_value=1,
            max_value=6,
            value=config.equipment.clay_traps,
            step=1
        )

def render_revenue_streams(config):
    """Render the revenue streams configuration"""
    st.subheader("Revenue Streams")
    st.write("Configure your workshop, class, and event offerings")
    
    # Workshops
    st.write("**Workshops**")
    config.workshops.enabled = st.checkbox(
        "Offer Workshops",
        value=config.workshops.enabled,
        help="Short pottery experiences for beginners"
    )
    
    if config.workshops.enabled:
        col1, col2 = st.columns(2)
        with col1:
            config.workshops.workshops_per_month = st.slider(
                "Workshops Per Month",
                min_value=0.0,
                max_value=12.0,
                value=config.workshops.workshops_per_month,
                step=0.5
            )
            
            config.workshops.workshop_fee = st.number_input(
                "Workshop Fee ($)",
                min_value=15.0,
                max_value=100.0,
                value=config.workshops.workshop_fee,
                step=5.0
            )
        
        with col2:
            config.workshops.avg_attendance = st.slider(
                "Average Attendance",
                min_value=4,
                max_value=20,
                value=int(config.workshops.avg_attendance),
                step=1
            )
            
            config.workshops.conversion_rate = st.slider(
                "Conversion to Membership (%)",
                min_value=0.0,
                max_value=50.0,
                value=config.workshops.conversion_rate * 100,
                step=2.5,
                help="Percentage of workshop attendees who become members"
            ) / 100
    
    # Classes
    st.write("**Structured Classes**")
    config.classes.enabled = st.checkbox(
        "Offer Classes",
        value=config.classes.enabled,
        help="Multi-week pottery courses"
    )
    
    if config.classes.enabled:
        col1, col2 = st.columns(2)
        with col1:
            config.classes.cohorts_per_month = st.slider(
                "New Classes Per Month",
                min_value=0.0,
                max_value=8.0,
                value=config.classes.cohorts_per_month,
                step=0.5
            )
            
            config.classes.class_price = st.number_input(
                "Class Series Price ($)",
                min_value=75.0,
                max_value=400.0,
                value=config.classes.class_price,
                step=10.0
            )
        
        with col2:
            config.classes.class_size_limit = st.slider(
                "Max Students Per Class",
                min_value=4,
                max_value=16,
                value=config.classes.class_size_limit,
                step=1
            )
            
            config.classes.conversion_rate = st.slider(
                "Class Conversion to Membership (%)",
                min_value=0.0,
                max_value=60.0,
                value=config.classes.conversion_rate * 100,
                step=2.5
            ) / 100
    
    # Events
    st.write("**Special Events**")
    config.events.enabled = st.checkbox(
        "Host Events",
        value=config.events.enabled,
        help="Paint nights, parties, corporate events"
    )
    
    if config.events.enabled:
        col1, col2 = st.columns(2)
        with col1:
            config.events.base_events_per_month = st.slider(
                "Average Events Per Month",
                min_value=0.0,
                max_value=10.0,
                value=config.events.base_events_per_month,
                step=0.5
            )
            
            config.events.ticket_price = st.number_input(
                "Event Ticket Price ($)",
                min_value=15.0,
                max_value=75.0,
                value=config.events.ticket_price,
                step=5.0
            )
        
        with col2:
            min_attendees = st.number_input(
                "Min Attendees",
                min_value=4,
                max_value=30,
                value=config.events.attendee_range[0],
                step=1
            )
            
            max_attendees = st.number_input(
                "Max Attendees",
                min_value=min_attendees,
                max_value=50,
                value=config.events.attendee_range[1],
                step=1
            )
            
            config.events.attendee_range = [int(min_attendees), int(max_attendees)]

def render_market_conditions(config):
    """Render market conditions and economic environment"""
    st.subheader("Market Conditions")
    st.write("How your studio responds to market changes and economic conditions")
    
    show_help_text("""
    **Price Sensitivity**: How much demand changes when you change prices.
    - Join Sensitivity: -1.5 means 10% price increase leads to 15% fewer new members
    - Churn Sensitivity: 1.2 means 10% price increase leads to 12% more members leaving
    
    **Economic Stress**: How often economic downturns affect your business.
    - 5% = Normal times (recession every 20 months on average)
    - 15% = Uncertain times 
    - 30% = Recession conditions
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pricing Response**")
        config.market.reference_price = st.number_input(
            "Competitor Average Price ($)",
            min_value=30.0,
            max_value=150.0,
            value=config.market.reference_price,
            step=5.0,
            help="What similar studios charge for monthly membership"
        )
        
        join_elasticity = st.slider(
            "Price Sensitivity - New Members",
            min_value=-3.0,
            max_value=-0.5,
            value=config.market.join_price_elasticity,
            step=0.1,
            help="How sensitive potential members are to your pricing"
        )
        config.market.join_price_elasticity = join_elasticity
        
        churn_elasticity = st.slider(
            "Price Sensitivity - Retention",
            min_value=0.2,
            max_value=2.5,
            value=config.market.churn_price_elasticity,
            step=0.1,
            help="How pricing affects member retention"
        )
        config.market.churn_price_elasticity = churn_elasticity
    
    with col2:
        st.write("**Economic Environment**")
        economic_stress = st.select_slider(
            "Economic Stress Level",
            options=[
                ("Normal", 0.05),
                ("Moderate", 0.08),
                ("Uncertain", 0.12),
                ("Stressed", 0.18),
                ("Recession", 0.30)
            ],
            value=("Moderate", 0.08),
            format_func=lambda x: x[0],
            help="How often economic stress affects your business"
        )
        config.economic.downturn_prob_per_month = economic_stress[1]
        
        config.economic.downturn_join_multiplier = st.slider(
            "Economic Stress Impact on New Members",
            min_value=0.3,
            max_value=1.0,
            value=config.economic.downturn_join_multiplier,
            step=0.05,
            help="Multiplier for new member rate during economic stress"
        )
        
        config.economic.downturn_churn_multiplier = st.slider(
            "Economic Stress Impact on Churn",
            min_value=1.0,
            max_value=2.5,
            value=config.economic.downturn_churn_multiplier,
            step=0.05,
            help="Multiplier for member churn during economic stress"
        )

def render_simulation_controls(config):
    """Render simulation controls"""
    st.subheader("Simulation Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config.simulation.time_horizon_months = st.selectbox(
            "Time Horizon",
            options=[12, 24, 36, 48, 60],
            index=2,  # Default to 36 months
            format_func=lambda x: f"{x} months ({x//12} years)"
        )
        
        config.simulation.num_simulations = st.selectbox(
            "Number of Simulations",
            options=[25, 50, 100, 200, 500],
            index=2,  # Default to 100
            help="More simulations = more accurate results but slower"
        )

def create_results_visualizations(results):
    """Create visualizations for simulation results"""
    
    # Summary metrics in columns
    summary = results.summary
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Survival Rate",
            f"{summary.survival_probability:.1%}",
            help="Probability of staying in business"
        )
        st.metric(
            "Break-even Month",
            f"{summary.break_even_month:.0f}" if not np.isnan(summary.break_even_month) else "N/A",
            help="When you start making profit"
        )
    
    with col2:
        st.metric(
            "Final Cash",
            f"${summary.median_final_cash:,.0f}",
            help="Median cash at end of simulation"
        )
        st.metric(
            "Cash Runway",
            f"{summary.cash_runway_months:.1f} months",
            help="How long your cash lasts in median scenario"
        )
    
    with col3:
        st.metric(
            "Monthly Revenue",
            f"${summary.median_monthly_revenue:,.0f}",
            help="Median monthly revenue"
        )
        st.metric(
            "Monthly Profit",
            f"${summary.median_monthly_profit:,.0f}",
            help="Median monthly profit"
        )
    
    with col4:
        st.metric(
            "Final Members",
            f"{summary.final_member_count:.0f}",
            help="Median member count at end"
        )
        if not np.isnan(summary.debt_service_coverage_ratio):
            st.metric(
                "Loan Health (DSCR)",
                f"{summary.debt_service_coverage_ratio:.2f}",
                help="Debt Service Coverage Ratio (>1.25 is healthy)"
            )
    
    # Cash flow chart
    st.subheader("Cash Flow Over Time")
    
    try:
        cash_timeseries = results.get_monthly_timeseries('cash_balance')
        
        fig = go.Figure()
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=cash_timeseries['month'],
            y=cash_timeseries['p90'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=cash_timeseries['month'],
            y=cash_timeseries['p10'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='10th-90th percentile',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=cash_timeseries['month'],
            y=cash_timeseries['p50'],
            mode='lines',
            name='Median',
            line=dict(color='blue', width=3)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", 
                     annotation_text="Insolvency Line")
        
        fig.update_layout(
            title="Cash Balance Projections",
            xaxis_title="Month",
            yaxis_title="Cash Balance ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not create cash flow chart: {e}")
    
    # Risk analysis
    st.subheader("Risk Analysis")
    
    try:
        risk_metrics = results.get_risk_analysis()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cash Flow Risks**")
            st.write(f"• Never go negative: {risk_metrics['cash_never_negative_prob']:.1%}")
            st.write(f"• Stay above $10k: {risk_metrics.get('cash_below_10k_prob', 0):.1%}")
            st.write(f"• Minimum cash (10th percentile): ${risk_metrics['minimum_cash_p10']:,.0f}")
        
        with col2:
            st.write("**Loan Servicing Risks**")
            if 'dscr_always_above_125_prob' in risk_metrics:
                st.write(f"• Always meet 1.25x DSCR: {risk_metrics['dscr_always_above_125_prob']:.1%}")
                st.write(f"• Never default (1.0x DSCR): {risk_metrics['dscr_always_above_100_prob']:.1%}")
                st.write(f"• Minimum DSCR (10th percentile): {risk_metrics['minimum_dscr_p10']:.2f}")
            else:
                st.write("DSCR data not available")
                
    except Exception as e:
        st.error(f"Could not create risk analysis: {e}")

def main():
    """Main Streamlit application"""
    
    st.title("🏺 Pottery Studio Business Simulator")
    st.write("**Financial modeling for pottery studio owners - Beta Version**")
    
    st.sidebar.title("Navigation")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Choose your experience level:",
        ["Quick Start", "Standard Planning", "Advanced Analysis"]
    )
    
    # Preset selection
    st.sidebar.subheader("Starting Point")
    preset_choice = st.sidebar.selectbox(
        "Choose a scenario preset:",
        options=list(SCENARIO_PRESETS.keys()),
        format_func=lambda x: x.title(),
        help="Starting assumptions for your analysis"
    )
    
    # Initialize configuration
    if 'config' not in st.session_state:
        st.session_state.config = create_preset_config(preset_choice)
    
    config = st.session_state.config
    
    # Show preset description
    st.sidebar.write(f"**{preset_choice.title()} Scenario**")
    st.sidebar.write(SCENARIO_PRESETS[preset_choice]["description"])
    
    if st.sidebar.button("Reset to Preset"):
        st.session_state.config = create_preset_config(preset_choice)
        st.experimental_rerun()
    
    # Parameter configuration based on mode
    if mode == "Quick Start":
        st.subheader("Quick Start - Essential Parameters Only")
        st.write("Set the most important parameters to get started quickly")
        
        col1, col2 = st.columns(2)
        with col1:
            config.business.monthly_rent = st.number_input(
                "Monthly Rent ($)", value=config.business.monthly_rent, step=100.0
            )
            config.business.membership_price = st.number_input(
                "Membership Price ($)", value=config.business.membership_price, step=5.0
            )
            config.business.max_members = st.number_input(
                "Studio Capacity", value=config.business.max_members, step=10
            )
        
        with col2:
            config.financing.loan_504_amount = st.number_input(
                "SBA 504 Loan ($)", value=config.financing.loan_504_amount, step=10000.0
            )
            config.financing.loan_7a_amount = st.number_input(
                "SBA 7(a) Loan ($)", value=config.financing.loan_7a_amount, step=5000.0
            )
            config.simulation.time_horizon_months = st.selectbox(
                "Time Horizon", options=[12, 24, 36], index=2, 
                format_func=lambda x: f"{x} months"
            )
    
    elif mode == "Standard Planning":
        st.subheader("Standard Planning - Complete Business Model")
        
        # Tabs for organized input
        tab1, tab2, tab3, tab4 = st.tabs([
            "Business & Financing", "Equipment & Revenue", "Market Conditions", "Simulation"
        ])
        
        with tab1:
            render_business_fundamentals(config)
            st.divider()
            render_financing_config(config)
        
        with tab2:
            render_equipment_config(config)
            st.divider()
            render_revenue_streams(config)
        
        with tab3:
            render_market_conditions(config)
        
        with tab4:
            render_simulation_controls(config)
    
    else:  # Advanced Analysis
        st.subheader("Advanced Analysis - Full Parameter Control")
        st.write("Complete access to all modeling parameters")
        
        with st.expander("Business Fundamentals", expanded=True):
            render_business_fundamentals(config)
        
        with st.expander("Financing Configuration", expanded=False):
            render_financing_config(config)
        
        with st.expander("Equipment Configuration", expanded=False):
            render_equipment_config(config)
        
        with st.expander("Revenue Streams", expanded=False):
            render_revenue_streams(config)
        
        with st.expander("Market Conditions", expanded=False):
            render_market_conditions(config)
        
        with st.expander("Simulation Controls", expanded=False):
            render_simulation_controls(config)
    
    # Validation and simulation
    st.divider()
    
    # Validate configuration
    errors = config.validate_all()
    if errors:
        st.error("Please fix the following issues before running simulation:")
        for section, section_errors in errors.items():
            st.write(f"**{section.title()}**: {', '.join(section_errors)}")
        return
    
    # Run simulation button
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation... This may take a minute."):
                try:
                    results = run_simulation(config)
                    st.session_state.results = results
                    st.success("Simulation completed!")
                except Exception as e:
                    st.error(f"Simulation failed: {e}")
                    return
    
    # Display results if available
    if 'results' in st.session_state:
        st.divider()
        st.header("Simulation Results")
        create_results_visualizations(st.session_state.results)
        
        # Download options
        st.subheader("Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Summary (CSV)"):
                summary_dict = st.session_state.results.summary.to_dict()
                summary_df = pd.DataFrame([summary_dict])
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="studio_simulation_summary.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Full Data (CSV)"):
                csv = st.session_state.results.raw_data.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="studio_simulation_detailed.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()