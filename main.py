#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Streamlit Parameter System - Exposes ALL model variables
"""

import io, json, re, zipfile
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot to fix rendering issues
import matplotlib.pyplot as plt
import seaborn as sns
from modular_simulator import get_default_cfg
from final_batch_adapter import run_original_once
from sba_export import export_to_sba_workbook
import os

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first available column from candidates list"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# COMPLETE PARAMETER SPECIFICATIONS - ALL MODEL VARIABLES
COMPLETE_PARAM_SPECS = {
    # =============================================================================
    # BUSINESS FUNDAMENTALS
    # =============================================================================
    "RENT": {
        "type": "float", "min": 1000, "max": 15000, "step": 100, "default": 3500,
        "label": "Monthly Base Rent ($)",
        "desc": "Fixed monthly rent payment for studio space. Directly impacts fixed costs and loan sizing calculations. Higher rent increases breakeven time and cash requirements.",
        "group": "business_fundamentals"
    },
    "RENT_GROWTH_PCT": {
        "type": "float", "min": 0.0, "max": 0.15, "step": 0.005, "default": 0.03,
        "label": "Annual Rent Growth Rate",
        "desc": "Yearly rent escalation as a decimal (0.03 = 3%). Compounds annually and affects long-term cash flow projections. Many leases include 2-4% annual increases.",
        "group": "business_fundamentals"
    },
    "OWNER_DRAW": {
        "type": "float", "min": 0, "max": 8000, "step": 100, "default": 2000,
        "label": "Owner Monthly Draw ($)",
        "desc": "Monthly cash withdrawal for owner living expenses. Reduces business cash flow and affects loan sizing. Set to 0 if owner takes no regular draw.",
        "group": "business_fundamentals"
    },
    "OWNER_DRAW_START_MONTH": {
        "type": "int", "min": 1, "max": 24, "step": 1, "default": 1,
        "label": "Owner Draw Start Month",
        "desc": "Month when owner draw payments begin (1-based). Allows deferring owner compensation during startup phase to preserve cash.",
        "group": "business_fundamentals"
    },
    "OWNER_DRAW_END_MONTH": {
        "type": "int", "min": 1, "max": 60, "step": 1, "default": 12,
        "label": "Owner Draw End Month (None=60)",
        "desc": "Last month of owner draw payments. Enter 60 for unlimited. Useful for modeling temporary owner sacrifice during startup.",
        "group": "business_fundamentals"
    },
    "OWNER_STIPEND_MONTHS": {
        "type": "int", "min": 0, "max": 60, "step": 1, "default": 12,
        "label": "Owner Stipend Duration (months)",
        "desc": "Total months of owner draw to reserve in cash planning. Even if draw window is longer, only this many months are included in loan sizing.",
        "group": "business_fundamentals"
    },
    
    # =============================================================================
    # MEMBER PRICING & ELASTICITY
    # =============================================================================
    "PRICE": {
        "type": "float", "min": 80, "max": 400, "step": 5, "default": 175,
        "label": "Monthly Membership Price ($)",
        "desc": "Base monthly membership fee charged to all new members. Affects both revenue and member acquisition/retention through price elasticity effects.",
        "group": "pricing"
    },
    "REFERENCE_PRICE": {
        "type": "float", "min": 80, "max": 400, "step": 5, "default": 165,
        "label": "Market Reference Price ($)",
        "desc": "Competitive baseline price for elasticity calculations. If your price is above this, expect lower join rates and higher churn. Use local market research to set this.",
        "group": "pricing"
    },
    "JOIN_PRICE_ELASTICITY": {
        "type": "float", "min": -3.0, "max": 0.0, "step": 0.1, "default": -0.6,
        "label": "Join Price Elasticity",
        "desc": "How sensitive potential members are to pricing. -0.6 means 10% price increase reduces joins by 6%. More negative = more price sensitive market.",
        "group": "pricing"
    },
    "CHURN_PRICE_ELASTICITY": {
        "type": "float", "min": 0.0, "max": 2.0, "step": 0.1, "default": 0.3,
        "label": "Churn Price Elasticity", 
        "desc": "How pricing affects member retention. 0.3 means 10% price increase increases churn by 3%. Higher values = more price-sensitive retention.",
        "group": "pricing"
    },
    
    # =============================================================================
    # MEMBER ARCHETYPES & BEHAVIOR
    # =============================================================================
    "HOBBYIST_PROB": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.35,
        "label": "Hobbyist Mix %",
        "desc": "Fraction of new members who are hobbyists. Casual users with lower usage and higher churn. Affects revenue per member and capacity utilization.",
        "group": "member_behavior"
    },
    "COMMITTED_ARTIST_PROB": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.40,
        "label": "Committed Artist Mix %",
        "desc": "Fraction of new members who are committed artists. Regular users with moderate usage and churn. Core revenue base for most studios.",
        "group": "member_behavior"
    },
    "PRODUCTION_POTTER_PROB": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.10,
        "label": "Production Potter Mix %",
        "desc": "Fraction of new members who are production potters. Heavy users with low churn but high capacity consumption. Valuable but space-intensive.",
        "group": "member_behavior"
    },
    "SEASONAL_USER_PROB": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.15,
        "label": "Seasonal User Mix %",
        "desc": "Fraction of new members who are seasonal users. Irregular usage with very high churn. Often driven by gift memberships or temporary interest.",
        "group": "member_behavior"
    },
    
    # Churn rates by archetype
    "ARCHETYPE_CHURN_HOBBYIST": {
        "type": "float", "min": 0.01, "max": 0.30, "step": 0.005, "default": 0.049 * 0.95,
        "label": "Hobbyist Monthly Churn Rate",
        "desc": "Base monthly churn probability for hobbyist members. Modified by tenure, pricing, and economic conditions. Typical range 3-8% monthly.",
        "group": "member_behavior"
    },
    "ARCHETYPE_CHURN_COMMITTED_ARTIST": {
        "type": "float", "min": 0.01, "max": 0.30, "step": 0.005, "default": 0.049 * 0.80,
        "label": "Committed Artist Monthly Churn Rate",
        "desc": "Base monthly churn probability for committed artist members. Generally lower than hobbyists due to higher engagement.",
        "group": "member_behavior"
    },
    "ARCHETYPE_CHURN_PRODUCTION_POTTER": {
        "type": "float", "min": 0.01, "max": 0.30, "step": 0.005, "default": 0.049 * 0.65,
        "label": "Production Potter Monthly Churn Rate",
        "desc": "Base monthly churn probability for production potter members. Lowest churn due to business dependency on studio access.",
        "group": "member_behavior"
    },
    "ARCHETYPE_CHURN_SEASONAL_USER": {
        "type": "float", "min": 0.01, "max": 0.50, "step": 0.005, "default": 0.049 * 1.90,
        "label": "Seasonal User Monthly Churn Rate",
        "desc": "Base monthly churn probability for seasonal user members. Highest churn due to temporary or gift-based engagement.",
        "group": "member_behavior"
    },
    
    # Usage patterns by archetype
    "HOBBYIST_SESSIONS_PER_WEEK": {
        "type": "float", "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.0,
        "label": "Hobbyist Sessions/Week",
        "desc": "Average studio sessions per week for hobbyist members. Affects capacity utilization calculations and revenue from add-on services.",
        "group": "member_behavior"
    },
    "COMMITTED_ARTIST_SESSIONS_PER_WEEK": {
        "type": "float", "min": 0.1, "max": 5.0, "step": 0.1, "default": 1.5,
        "label": "Committed Artist Sessions/Week", 
        "desc": "Average studio sessions per week for committed artist members. Higher usage drives more clay sales and firing fees.",
        "group": "member_behavior"
    },
    "PRODUCTION_POTTER_SESSIONS_PER_WEEK": {
        "type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "default": 3.5,
        "label": "Production Potter Sessions/Week",
        "desc": "Average studio sessions per week for production potter members. Highest usage, may constrain capacity for other members.",
        "group": "member_behavior"
    },
    "SEASONAL_USER_SESSIONS_PER_WEEK": {
        "type": "float", "min": 0.1, "max": 5.0, "step": 0.1, "default": 0.75,
        "label": "Seasonal User Sessions/Week",
        "desc": "Average studio sessions per week for seasonal user members. Lower usage reflects casual engagement level.",
        "group": "member_behavior"
    },
    
    # Session duration by archetype
    "HOBBYIST_SESSION_HOURS": {
        "type": "float", "min": 0.5, "max": 8.0, "step": 0.1, "default": 1.7,
        "label": "Hobbyist Hours/Session",
        "desc": "Average hours per studio session for hobbyist members. Shorter sessions allow more members to use equipment during peak times.",
        "group": "member_behavior"
    },
    "COMMITTED_ARTIST_SESSION_HOURS": {
        "type": "float", "min": 0.5, "max": 8.0, "step": 0.1, "default": 2.75,
        "label": "Committed Artist Hours/Session",
        "desc": "Average hours per studio session for committed artist members. Longer sessions reflect deeper engagement with projects.",
        "group": "member_behavior"
    },
    "PRODUCTION_POTTER_SESSION_HOURS": {
        "type": "float", "min": 0.5, "max": 12.0, "step": 0.1, "default": 3.8,
        "label": "Production Potter Hours/Session",
        "desc": "Average hours per studio session for production potter members. Longest sessions due to commercial production needs.",
        "group": "member_behavior"
    },
    "SEASONAL_USER_SESSION_HOURS": {
        "type": "float", "min": 0.5, "max": 8.0, "step": 0.1, "default": 2.0,
        "label": "Seasonal User Hours/Session",
        "desc": "Average hours per studio session for seasonal user members. Moderate duration typical of casual engagement.",
        "group": "member_behavior"
    },
    
    # =============================================================================
    # CAPACITY & STATIONS
    # =============================================================================
    "MAX_MEMBERS": {
        "type": "int", "min": 20, "max": 500, "step": 5, "default": 77,
        "label": "Maximum Members (Hard Cap)",
        "desc": "Absolute maximum members the studio can accommodate. Based on physical space, storage, and operational constraints. Acts as hard limit on growth.",
        "group": "capacity"
    },
    "OPEN_HOURS_PER_WEEK": {
        "type": "int", "min": 20, "max": 168, "step": 4, "default": 112,
        "label": "Studio Open Hours/Week",
        "desc": "Total weekly hours studio is accessible to members. Affects capacity calculations - more hours = more member capacity for same equipment.",
        "group": "capacity"
    },
    "CAPACITY_DAMPING_BETA": {
        "type": "float", "min": 1.0, "max": 10.0, "step": 0.5, "default": 4.0,
        "label": "Capacity Damping Factor",
        "desc": "Controls how crowding reduces new member joins. Higher values = sharper drop in joins as studio gets crowded. 4.0 means severe impact near capacity.",
        "group": "capacity"
    },
    "UTILIZATION_CHURN_UPLIFT": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.25,
        "label": "Overcrowding Churn Multiplier",
        "desc": "Additional churn when studio is over capacity. 0.25 means 25% higher churn when 100% utilized. Models member frustration with crowding.",
        "group": "capacity"
    },
    
    # Station capacities
    "WHEELS_CAPACITY": {
        "type": "int", "min": 2, "max": 30, "step": 1, "default": 8,
        "label": "Pottery Wheels Count",
        "desc": "Number of pottery wheels available. Often the limiting factor for member capacity since wheels are used by all archetypes.",
        "group": "capacity"
    },
    "HANDBUILDING_CAPACITY": {
        "type": "int", "min": 2, "max": 50, "step": 1, "default": 6,
        "label": "Handbuilding Stations Count",
        "desc": "Number of handbuilding workstations. Used for sculpture, handbuilding, and surface decoration work.",
        "group": "capacity"
    },
    "GLAZE_CAPACITY": {
        "type": "int", "min": 2, "max": 20, "step": 1, "default": 6,
        "label": "Glazing Stations Count",
        "desc": "Number of glazing workstations. Bottleneck station in many studios since all fired work needs glazing.",
        "group": "capacity"
    },
    
    # Station utilization factors
    "WHEELS_ALPHA": {
        "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.80,
        "label": "Wheels Utilization Efficiency",
        "desc": "Fraction of wheel capacity actually usable (accounting for maintenance, setup time, etc.). 0.80 = 80% effective utilization.",
        "group": "capacity"
    },
    "HANDBUILDING_ALPHA": {
        "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.50,
        "label": "Handbuilding Utilization Efficiency",
        "desc": "Fraction of handbuilding capacity actually usable. Lower than wheels due to variable project sizes and cleanup time.",
        "group": "capacity"
    },
    "GLAZE_ALPHA": {
        "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.55,
        "label": "Glazing Utilization Efficiency",
        "desc": "Fraction of glazing capacity actually usable. Accounts for drying time, glaze prep, and safety procedures.",
        "group": "capacity"
    },
    
    # =============================================================================
    # MARKET DYNAMICS & ACQUISITION
    # =============================================================================
    "NO_ACCESS_POOL": {
        "type": "int", "min": 0, "max": 1000, "step": 10, "default": 20,
        "label": "No-Access Market Pool Size",
        "desc": "People in your market who have no current pottery access. Most motivated to join but need discovery. Size depends on local population.",
        "group": "market_dynamics"
    },
    "HOME_POOL": {
        "type": "int", "min": 0, "max": 1000, "step": 10, "default": 50,
        "label": "Home Studio Market Pool Size", 
        "desc": "People with home pottery setups. Less motivated to join due to existing access. May join for community, equipment, or firing access.",
        "group": "market_dynamics"
    },
    "COMMUNITY_POOL": {
        "type": "int", "min": 0, "max": 1000, "step": 10, "default": 70,
        "label": "Community Studio Market Pool Size",
        "desc": "People currently using other community studios. May switch if you offer better value/location/community. Existing pottery experience.",
        "group": "market_dynamics"
    },
    
    # Market inflows (replenishment)
    "NO_ACCESS_INFLOW": {
        "type": "int", "min": 0, "max": 50, "step": 1, "default": 3,
        "label": "No-Access Monthly Inflow",
        "desc": "New people entering the no-access pool each month (moved to area, developed interest, etc.). Sustains long-term member acquisition.",
        "group": "market_dynamics"
    },
    "HOME_INFLOW": {
        "type": "int", "min": 0, "max": 50, "step": 1, "default": 2,
        "label": "Home Studio Monthly Inflow",
        "desc": "People setting up home studios monthly. May eventually seek community/professional equipment. Usually pottery enthusiasts.",
        "group": "market_dynamics"
    },
    "COMMUNITY_INFLOW": {
        "type": "int", "min": 0, "max": 50, "step": 1, "default": 4,
        "label": "Community Studio Monthly Inflow",
        "desc": "People joining other studios monthly. Potential switchers if dissatisfied with current studio. Higher intent but harder to reach.",
        "group": "market_dynamics"
    },
    
    # Base join rates by pool
    "BASELINE_RATE_NO_ACCESS": {
        "type": "float", "min": 0.0, "max": 0.2, "step": 0.005, "default": 0.040,
        "label": "No-Access Monthly Join Rate",
        "desc": "Base probability that a no-access person joins per month. Modified by marketing, pricing, capacity, and economic factors.",
        "group": "market_dynamics"
    },
    "BASELINE_RATE_HOME": {
        "type": "float", "min": 0.0, "max": 0.1, "step": 0.005, "default": 0.010,
        "label": "Home Studio Monthly Join Rate",
        "desc": "Base probability that a home studio person joins per month. Lower due to existing access, but may join for community/equipment.",
        "group": "market_dynamics"
    },
    "BASELINE_RATE_COMMUNITY": {
        "type": "float", "min": 0.0, "max": 0.3, "step": 0.005, "default": 0.100,
        "label": "Community Studio Monthly Join Rate",
        "desc": "Base probability that a person at another studio switches per month. Higher due to existing pottery commitment and switching motivations.",
        "group": "market_dynamics"
    },
    
    # Word of mouth and referrals
    "WOM_Q": {
        "type": "float", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.60,
        "label": "Word-of-Mouth Amplification Factor",
        "desc": "How much word-of-mouth boosts join rates. 0.6 with 60 members near saturation doubles join rates. Higher = stronger community effect.",
        "group": "market_dynamics"
    },
    "WOM_SATURATION": {
        "type": "int", "min": 20, "max": 200, "step": 5, "default": 60,
        "label": "Word-of-Mouth Saturation Point",
        "desc": "Member count where WOM effect peaks. Beyond this, additional members provide diminishing word-of-mouth returns. Market size dependent.",
        "group": "market_dynamics"
    },
    "REFERRAL_RATE_PER_MEMBER": {
        "type": "float", "min": 0.0, "max": 0.3, "step": 0.01, "default": 0.06,
        "label": "Monthly Referral Rate per Member",
        "desc": "Probability each member generates a referral per month. 0.06 = 6% chance per member monthly. Drives organic growth through direct recommendations.",
        "group": "market_dynamics"
    },
    "REFERRAL_CONV": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.22,
        "label": "Referral Conversion Rate",
        "desc": "Probability that a referral becomes a member. Higher than cold prospects due to friend recommendation and fit pre-screening.",
        "group": "market_dynamics"
    },
    
    # Awareness and adoption
    "AWARENESS_RAMP_MONTHS": {
        "type": "int", "min": 1, "max": 24, "step": 1, "default": 4,
        "label": "Awareness Ramp Duration (months)",
        "desc": "Months to reach full market awareness. Longer ramp = slower initial growth but models realistic awareness building in new markets.",
        "group": "market_dynamics"
    },
    "AWARENESS_RAMP_START_MULT": {
        "type": "float", "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.5,
        "label": "Starting Awareness Level",
        "desc": "Market awareness at launch as fraction of eventual level. 0.5 = 50% awareness at start, ramping to 100% over ramp period.",
        "group": "market_dynamics"
    },
    "AWARENESS_RAMP_END_MULT": {
        "type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0,
        "label": "Peak Awareness Level", 
        "desc": "Maximum market awareness as multiplier. 1.0 = normal market penetration, >1.0 = exceptional awareness (strong marketing/PR).",
        "group": "market_dynamics"
    },
    "ADOPTION_SIGMA": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.20,
        "label": "Adoption Noise Factor",
        "desc": "Random variation in monthly adoption (lognormal sigma). 0.20 adds realistic month-to-month variation. Higher = more volatile growth.",
        "group": "market_dynamics"
    },
    
    # Community studio switching
    "CLASS_TERM_MONTHS": {
        "type": "int", "min": 1, "max": 12, "step": 1, "default": 3,
        "label": "Class Term Length (months)",
        "desc": "How often community studio members can switch (class graduation cycles). 3 months = quarterly switching opportunities.",
        "group": "market_dynamics"
    },
    "CS_UNLOCK_FRACTION_PER_TERM": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.25,
        "label": "Community Studio Unlock Rate",
        "desc": "Fraction of remaining CS pool eligible to switch each term. 0.25 = 25% of remaining pool becomes available every term cycle.",
        "group": "market_dynamics"
    },
    
    # Onboarding capacity
    "MAX_ONBOARDINGS_PER_MONTH": {
        "type": "int", "min": 1, "max": 100, "step": 1, "default": 10,
        "label": "Max New Members/Month",
        "desc": "Operational limit on monthly new member onboarding. Accounts for orientation capacity, key cutting, etc. None = unlimited.",
        "group": "operations"
    },
    
    # =============================================================================
    # ECONOMIC ENVIRONMENT
    # =============================================================================
    "DOWNTURN_PROB_PER_MONTH": {
        "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.05,
        "label": "Monthly Economic Stress Probability",
        "desc": "Probability of economic stress in any given month. 0.05 = 5% monthly chance. During stress, join/churn rates change according to multipliers below.",
        "group": "economic_environment"
    },
    "DOWNTURN_JOIN_MULT": {
        "type": "float", "min": 0.1, "max": 2.0, "step": 0.05, "default": 1.0,
        "label": "Economic Stress Join Multiplier",
        "desc": "Join rate multiplier during economic stress months. 0.65 = 35% reduction in joins during downturns. <1.0 = people delay discretionary spending.",
        "group": "economic_environment"
    },
    "DOWNTURN_CHURN_MULT": {
        "type": "float", "min": 0.1, "max": 3.0, "step": 0.05, "default": 1.0,
        "label": "Economic Stress Churn Multiplier", 
        "desc": "Churn rate multiplier during economic stress months. 1.50 = 50% increase in churn during downturns. >1.0 = people cut discretionary spending.",
        "group": "economic_environment"
    },
    
    # =============================================================================
    # SEASONALITY
    # =============================================================================
    "SEASONALITY_JAN": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.1, "label": "January Seasonality", "desc": "January activity multiplier vs average month. >1.0 = above average (New Year resolutions)", "group": "seasonality"},
    "SEASONALITY_FEB": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.2, "label": "February Seasonality", "desc": "February activity multiplier. Often strong due to Valentine's Day pottery gifts", "group": "seasonality"},
    "SEASONALITY_MAR": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.3, "label": "March Seasonality", "desc": "March activity multiplier. Spring renewal and Mother's Day prep", "group": "seasonality"},
    "SEASONALITY_APR": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.4, "label": "April Seasonality", "desc": "April activity multiplier. Peak spring activity", "group": "seasonality"},
    "SEASONALITY_MAY": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.3, "label": "May Seasonality", "desc": "May activity multiplier. Mother's Day and spring continues", "group": "seasonality"},
    "SEASONALITY_JUN": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 0.9, "label": "June Seasonality", "desc": "June activity multiplier. Summer vacation season begins", "group": "seasonality"},
    "SEASONALITY_JUL": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 0.8, "label": "July Seasonality", "desc": "July activity multiplier. Peak vacation season", "group": "seasonality"},
    "SEASONALITY_AUG": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 0.85, "label": "August Seasonality", "desc": "August activity multiplier. Late summer, back-to-school prep", "group": "seasonality"},
    "SEASONALITY_SEP": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.3, "label": "September Seasonality", "desc": "September activity multiplier. Back-to-school and fall activity surge", "group": "seasonality"},
    "SEASONALITY_OCT": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.4, "label": "October Seasonality", "desc": "October activity multiplier. Peak fall activity and holiday prep", "group": "seasonality"},
    "SEASONALITY_NOV": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.2, "label": "November Seasonality", "desc": "November activity multiplier. Holiday gift making season", "group": "seasonality"},
    "SEASONALITY_DEC": {"type": "float", "min": 0.5, "max": 2.0, "step": 0.05, "default": 1.0, "label": "December Seasonality", "desc": "December activity multiplier. Holiday season but also vacation time", "group": "seasonality"},
    
    # =============================================================================
    # REVENUE: CLAY AND FIRING
    # =============================================================================
    "RETAIL_CLAY_PRICE_PER_BAG": {
        "type": "float", "min": 15.0, "max": 50.0, "step": 1.0, "default": 25.0,
        "label": "Retail Clay Price ($/bag)",
        "desc": "Price charged to members per 25lb clay bag. Key add-on revenue stream. Typically marked up 40-60% over wholesale cost.",
        "group": "clay_firing_revenue"
    },
    "WHOLESALE_CLAY_COST_PER_BAG": {
        "type": "float", "min": 8.0, "max": 30.0, "step": 0.25, "default": 16.75,
        "label": "Wholesale Clay Cost ($/bag)",
        "desc": "Cost of clay per 25lb bag from supplier. Direct cost of goods sold. Affects profit margin on clay sales to members.",
        "group": "clay_firing_revenue"
    },
    
    # Clay usage by archetype (low, typical, high bags per month)
    "HOBBYIST_CLAY_LOW": {
        "type": "float", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.25,
        "label": "Hobbyist Clay Usage - Low (bags/month)",
        "desc": "Minimum monthly clay consumption for hobbyist members. Part of triangular distribution modeling usage variation.",
        "group": "clay_firing_revenue"
    },
    "HOBBYIST_CLAY_TYPICAL": {
        "type": "float", "min": 0.1, "max": 3.0, "step": 0.1, "default": 0.5,
        "label": "Hobbyist Clay Usage - Typical (bags/month)",
        "desc": "Most common monthly clay consumption for hobbyist members. Peak of triangular distribution.",
        "group": "clay_firing_revenue"
    },
    "HOBBYIST_CLAY_HIGH": {
        "type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "default": 1.0,
        "label": "Hobbyist Clay Usage - High (bags/month)",
        "desc": "Maximum monthly clay consumption for hobbyist members. Upper bound of triangular distribution.",
        "group": "clay_firing_revenue"
    },
    
    "COMMITTED_ARTIST_CLAY_LOW": {
        "type": "float", "min": 0.5, "max": 3.0, "step": 0.1, "default": 1.0,
        "label": "Committed Artist Clay Usage - Low (bags/month)",
        "desc": "Minimum monthly clay consumption for committed artist members.",
        "group": "clay_firing_revenue"
    },
    "COMMITTED_ARTIST_CLAY_TYPICAL": {
        "type": "float", "min": 0.5, "max": 4.0, "step": 0.1, "default": 1.5,
        "label": "Committed Artist Clay Usage - Typical (bags/month)",
        "desc": "Most common monthly clay consumption for committed artist members.",
        "group": "clay_firing_revenue"
    },
    "COMMITTED_ARTIST_CLAY_HIGH": {
        "type": "float", "min": 1.0, "max": 6.0, "step": 0.1, "default": 2.0,
        "label": "Committed Artist Clay Usage - High (bags/month)",
        "desc": "Maximum monthly clay consumption for committed artist members.",
        "group": "clay_firing_revenue"
    },
    
    "PRODUCTION_POTTER_CLAY_LOW": {
        "type": "float", "min": 1.0, "max": 5.0, "step": 0.1, "default": 2.0,
        "label": "Production Potter Clay Usage - Low (bags/month)",
        "desc": "Minimum monthly clay consumption for production potter members.",
        "group": "clay_firing_revenue"
    },
    "PRODUCTION_POTTER_CLAY_TYPICAL": {
        "type": "float", "min": 1.5, "max": 6.0, "step": 0.1, "default": 2.5,
        "label": "Production Potter Clay Usage - Typical (bags/month)",
        "desc": "Most common monthly clay consumption for production potter members.",
        "group": "clay_firing_revenue"
    },
    "PRODUCTION_POTTER_CLAY_HIGH": {
        "type": "float", "min": 2.0, "max": 10.0, "step": 0.1, "default": 3.0,
        "label": "Production Potter Clay Usage - High (bags/month)",
        "desc": "Maximum monthly clay consumption for production potter members.",
        "group": "clay_firing_revenue"
    },
    
    "SEASONAL_USER_CLAY_LOW": {
        "type": "float", "min": 0.1, "max": 2.0, "step": 0.1, "default": 0.25,
        "label": "Seasonal User Clay Usage - Low (bags/month)",
        "desc": "Minimum monthly clay consumption for seasonal user members.",
        "group": "clay_firing_revenue"
    },
    "SEASONAL_USER_CLAY_TYPICAL": {
        "type": "float", "min": 0.1, "max": 3.0, "step": 0.1, "default": 0.5,
        "label": "Seasonal User Clay Usage - Typical (bags/month)",
        "desc": "Most common monthly clay consumption for seasonal user members.",
        "group": "clay_firing_revenue"
    },
    "SEASONAL_USER_CLAY_HIGH": {
        "type": "float", "min": 0.5, "max": 5.0, "step": 0.1, "default": 1.0,
        "label": "Seasonal User Clay Usage - High (bags/month)",
        "desc": "Maximum monthly clay consumption for seasonal user members.",
        "group": "clay_firing_revenue"
    },
    
    # =============================================================================
    # REVENUE: WORKSHOPS
    # =============================================================================
    "WORKSHOPS_ENABLED": {
        "type": "bool", "default": True,
        "label": "Enable Workshop Revenue Stream",
        "desc": "Whether studio offers short pottery workshops for beginners. Key revenue and member acquisition channel for many studios.",
        "group": "workshops"
    },
    "WORKSHOPS_PER_MONTH": {
        "type": "float", "min": 0.0, "max": 20.0, "step": 0.5, "default": 2.0,
        "label": "Workshops per Month",
        "desc": "Average number of workshops offered monthly. More workshops = more revenue but requires instructor time and capacity.",
        "group": "workshops"
    },
    "WORKSHOP_AVG_ATTENDANCE": {
        "type": "int", "min": 1, "max": 30, "step": 1, "default": 10,
        "label": "Average Workshop Attendance",
        "desc": "Typical number of participants per workshop. Limited by space and instructor capacity.",
        "group": "workshops"
    },
    "WORKSHOP_FEE": {
        "type": "float", "min": 20.0, "max": 150.0, "step": 5.0, "default": 75.0,
        "label": "Workshop Fee per Person ($)",
        "desc": "Price charged per workshop participant. Key revenue driver - should cover materials, instructor time, and profit margin.",
        "group": "workshops"
    },
    "WORKSHOP_COST_PER_EVENT": {
        "type": "float", "min": 0.0, "max": 500.0, "step": 10.0, "default": 50.0,
        "label": "Workshop Variable Cost per Event ($)",
        "desc": "Direct costs per workshop: instructor pay, materials, cleanup. Subtracted from gross revenue to get net contribution.",
        "group": "workshops"
    },
    "WORKSHOP_CONV_RATE": {
        "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.12,
        "label": "Workshop to Member Conversion Rate",
        "desc": "Fraction of workshop participants who become members. Key metric - workshops as member acquisition funnel.",
        "group": "workshops"
    },
    "WORKSHOP_CONV_LAG_MO": {
        "type": "int", "min": 0, "max": 6, "step": 1, "default": 1,
        "label": "Workshop Conversion Lag (months)",
        "desc": "Months between workshop participation and membership signup. Accounts for decision time and class schedules.",
        "group": "workshops"
    },
    
    # =============================================================================
    # REVENUE: CLASSES
    # =============================================================================
    "CLASSES_ENABLED": {
        "type": "bool", "default": True,
        "label": "Enable Class Revenue Stream",
        "desc": "Whether studio offers multi-week pottery courses. Higher revenue per participant but requires structured curriculum.",
        "group": "classes"
    },
    "CLASSES_CALENDAR_MODE": {
        "type": "select", "options": ["monthly", "semester"], "default": "semester",
        "label": "Class Schedule Type",
        "desc": "Monthly = continuous rolling classes. Semester = structured terms with breaks. Affects cash flow timing and member acquisition patterns.",
        "group": "classes"
    },
    "CLASS_COHORTS_PER_MONTH": {
        "type": "int", "min": 0, "max": 10, "step": 1, "default": 2,
        "label": "Class Cohorts per Month/Term",
        "desc": "Number of class groups starting per period. More cohorts = more revenue but requires instructor capacity.",
        "group": "classes"
    },
    "CLASS_CAP_PER_COHORT": {
        "type": "int", "min": 3, "max": 20, "step": 1, "default": 10,
        "label": "Students per Class",
        "desc": "Maximum students per class cohort. Limited by instruction quality and workspace capacity.",
        "group": "classes"
    },
    "CLASS_PRICE": {
        "type": "float", "min": 100.0, "max": 1000.0, "step": 25.0, "default": 600.0,
        "label": "Class Series Price ($)",
        "desc": "Tuition for complete multi-week course. Major revenue stream - should cover instructor costs, materials, and profit.",
        "group": "classes"
    },
    "CLASS_FILL_MEAN": {
        "type": "float", "min": 0.3, "max": 1.0, "step": 0.05, "default": 0.85,
        "label": "Average Class Fill Rate",
        "desc": "Typical fraction of class capacity that actually enrolls. 0.85 = 85% average enrollment. Accounts for no-shows and cancellations.",
        "group": "classes"
    },
    "CLASS_COST_PER_STUDENT": {
        "type": "float", "min": 10.0, "max": 100.0, "step": 5.0, "default": 40.0,
        "label": "Variable Cost per Student ($)",
        "desc": "Materials and supplies cost per class student over full course. Clay, glazes, firing costs, handouts, etc.",
        "group": "classes"
    },
    "CLASS_INSTR_RATE_PER_HR": {
        "type": "float", "min": 15.0, "max": 100.0, "step": 2.5, "default": 30.0,
        "label": "Instructor Hourly Rate ($)",
        "desc": "Compensation for class instructor per hour. Major cost component for class programs.",
        "group": "classes"
    },
    "CLASS_HOURS_PER_COHORT": {
        "type": "float", "min": 6.0, "max": 40.0, "step": 1.0, "default": 18.0,
        "label": "Total Hours per Class Series",
        "desc": "Total instructor hours per complete class (e.g., 6 weeks × 3 hours = 18). Affects instructor costs.",
        "group": "classes"
    },
    "CLASS_CONV_RATE": {
        "type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.12,
        "label": "Class to Member Conversion Rate",
        "desc": "Fraction of class students who become members. Higher than workshop conversion due to deeper engagement.",
        "group": "classes"
    },
    "CLASS_CONV_LAG_MO": {
        "type": "int", "min": 0, "max": 6, "step": 1, "default": 1,
        "label": "Class Conversion Lag (months)",
        "desc": "Months between class completion and membership signup. Often immediate as students are already engaged.",
        "group": "classes"
    },
    "CLASS_EARLY_CHURN_MULT": {
        "type": "float", "min": 0.1, "max": 1.5, "step": 0.05, "default": 0.8,
        "label": "Class Convert Early Churn Multiplier",
        "desc": "Churn rate modifier for class converts in first 3-6 months. <1.0 = lower churn due to formal introduction to pottery.",
        "group": "classes"
    },
    
    # Class semester scheduling
    "CLASS_SEMESTER_LENGTH_MONTHS": {
        "type": "int", "min": 1, "max": 6, "step": 1, "default": 3,
        "label": "Semester Length (months)",
        "desc": "Duration of each semester in months. Only relevant if using semester scheduling mode.",
        "group": "classes"
    },
    
    # =============================================================================
    # REVENUE: EVENTS
    # =============================================================================
    "EVENTS_ENABLED": {
        "type": "bool", "default": True,
        "label": "Enable Event Revenue Stream",
        "desc": "Whether studio hosts public events (paint-a-pot, sip & paint, parties). Popular revenue stream with good margins.",
        "group": "events"
    },
    "BASE_EVENTS_PER_MONTH_LAMBDA": {
        "type": "float", "min": 0.0, "max": 20.0, "step": 0.5, "default": 3.0,
        "label": "Base Events per Month (λ)",
        "desc": "Average events per month (Poisson distribution). Seasonal multipliers apply on top of this base rate.",
        "group": "events"
    },
    "EVENTS_MAX_PER_MONTH": {
        "type": "int", "min": 1, "max": 30, "step": 1, "default": 4,
        "label": "Maximum Events per Month",
        "desc": "Hard cap on monthly events due to staff/space constraints. Prevents unrealistic event counts during high-demand periods.",
        "group": "events"
    },
    "TICKET_PRICE": {
        "type": "float", "min": 30.0, "max": 200.0, "step": 5.0, "default": 75.0,
        "label": "Event Ticket Price ($)",
        "desc": "Price per event participant. Should cover materials, staff time, and generate profit. Market-dependent pricing.",
        "group": "events"
    },
    "ATTENDEES_PER_EVENT_RANGE": {
        "type": "text", "default": "[8, 10, 12]",
        "label": "Event Attendance Range (JSON list)",
        "desc": "Possible attendance numbers per event as JSON list. Model randomly selects from these values each event.",
        "group": "events"
    },
    "EVENT_MUG_COST_RANGE": {
        "type": "text", "default": "[4.5, 7.5]",
        "label": "Event Mug Cost Range (JSON [min, max])",
        "desc": "Cost range for bisque mugs per participant as JSON [min, max]. Random cost drawn from uniform distribution.",
        "group": "events"
    },
    "EVENT_CONSUMABLES_PER_PERSON": {
        "type": "float", "min": 1.0, "max": 20.0, "step": 0.5, "default": 2.5,
        "label": "Event Consumables Cost per Person ($)",
        "desc": "Cost of supplies per participant: glazes, brushes, cleanup materials, packaging, etc.",
        "group": "events"
    },
    "EVENT_STAFF_RATE_PER_HOUR": {
        "type": "float", "min": 0.0, "max": 50.0, "step": 1.0, "default": 22.0,
        "label": "Event Staff Hourly Rate ($)",
        "desc": "Hourly compensation for event staff/instructor. Set to 0 if events are run by owner with no additional cost.",
        "group": "events"
    },
    "EVENT_HOURS_PER_EVENT": {
        "type": "float", "min": 1.0, "max": 8.0, "step": 0.5, "default": 2.0,
        "label": "Staff Hours per Event",
        "desc": "Staff time per event including setup, instruction, and cleanup. Multiplied by hourly rate for labor cost.",
        "group": "events"
    },
    
    # =============================================================================
    # REVENUE: DESIGNATED STUDIOS
    # =============================================================================
    "DESIGNATED_STUDIO_COUNT": {
        "type": "int", "min": 0, "max": 10, "step": 1, "default": 2,
        "label": "Number of Designated Studios",
        "desc": "Private workspace rentals for serious artists. Premium revenue stream with dedicated space allocation.",
        "group": "designated_studios"
    },
    "DESIGNATED_STUDIO_PRICE": {
        "type": "float", "min": 100.0, "max": 1000.0, "step": 25.0, "default": 300.0,
        "label": "Designated Studio Monthly Price ($)",
        "desc": "Monthly rental fee per designated studio. Premium pricing for private workspace and storage.",
        "group": "designated_studios"
    },
    "DESIGNATED_STUDIO_BASE_OCCUPANCY": {
        "type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3,
        "label": "Designated Studio Occupancy Rate",
        "desc": "Average fraction of designated studios occupied. Lower occupancy = harder to fill premium spaces.",
        "group": "designated_studios"
    },
    
    # =============================================================================
    # OPERATING COSTS: FIXED
    # =============================================================================
    "INSURANCE_COST": {
        "type": "float", "min": 50.0, "max": 500.0, "step": 10.0, "default": 75.0,
        "label": "Monthly Insurance Cost ($)",
        "desc": "General liability and property insurance. Required for most leases and essential for pottery studio operations.",
        "group": "fixed_costs"
    },
    "GLAZE_COST_PER_MONTH": {
        "type": "float", "min": 200.0, "max": 2000.0, "step": 50.0, "default": 833.33,
        "label": "Monthly Glaze Cost ($)",
        "desc": "Glazes, underglazes, and finishing materials. Fixed cost since members use unlimited glazes. Major expense item.",
        "group": "fixed_costs"
    },
    "HEATING_COST_WINTER": {
        "type": "float", "min": 100.0, "max": 1500.0, "step": 25.0, "default": 450.0,
        "label": "Winter Monthly Heating ($)",
        "desc": "Heating costs during cold months (Oct-Mar). Higher for pottery studios due to large spaces and kiln heat loss.",
        "group": "fixed_costs"
    },
    "HEATING_COST_SUMMER": {
        "type": "float", "min": 0.0, "max": 500.0, "step": 10.0, "default": 30.0,
        "label": "Summer Monthly Heating ($)",
        "desc": "Minimal heating costs during warm months (Apr-Sep). May be just hot water heater and minimal space heating.",
        "group": "fixed_costs"
    },
    
    # =============================================================================
    # OPERATING COSTS: VARIABLE
    # =============================================================================
    "COST_PER_KWH": {
        "type": "float", "min": 0.08, "max": 0.50, "step": 0.01, "default": 0.2182,
        "label": "Electricity Rate ($/kWh)",
        "desc": "Local electricity rate including all fees and taxes. Check recent utility bills. Kilns are major electricity consumers.",
        "group": "variable_costs"
    },
    "WATER_COST_PER_GALLON": {
        "type": "float", "min": 0.005, "max": 0.05, "step": 0.002, "default": 0.02,
        "label": "Water Cost ($/gallon)",
        "desc": "Water and sewer costs per gallon. Pottery uses significant water for clay prep and cleanup.",
        "group": "variable_costs"
    },
    "GALLONS_PER_BAG_CLAY": {
        "type": "float", "min": 0.5, "max": 3.0, "step": 0.1, "default": 1.0,
        "label": "Water per Clay Bag (gallons)",
        "desc": "Water consumption per 25lb clay bag for mixing and cleanup. Varies by clay type and studio practices.",
        "group": "variable_costs"
    },
    
    # Kiln electricity usage
    "KWH_PER_FIRING_KMT1027": {
        "type": "float", "min": 40.0, "max": 120.0, "step": 5.0, "default": 75.0,
        "label": "kWh per Firing - Kiln 1 (KMT1027)",
        "desc": "Electricity consumption per firing cycle for smaller kiln. Varies by firing temperature and duration.",
        "group": "variable_costs"
    },
    "KWH_PER_FIRING_KMT1427": {
        "type": "float", "min": 60.0, "max": 180.0, "step": 5.0, "default": 110.0,
        "label": "kWh per Firing - Kiln 2 (KMT1427)",
        "desc": "Electricity consumption per firing cycle for larger kiln. Higher capacity but more energy per firing.",
        "group": "variable_costs"
    },
    
    # Kiln scheduling
    "DYNAMIC_FIRINGS": {
        "type": "bool", "default": True,
        "label": "Dynamic Firing Schedule",
        "desc": "Whether firing frequency adjusts based on member count. True = more members trigger more firings. False = fixed schedule.",
        "group": "variable_costs"
    },
    "BASE_FIRINGS_PER_MONTH": {
        "type": "int", "min": 2, "max": 30, "step": 1, "default": 10,
        "label": "Base Firings per Month",
        "desc": "Firing frequency at reference member count. Used for scaling if dynamic firings enabled, or as fixed rate if disabled.",
        "group": "variable_costs"
    },
    "REFERENCE_MEMBERS_FOR_BASE_FIRINGS": {
        "type": "int", "min": 5, "max": 50, "step": 1, "default": 12,
        "label": "Reference Member Count for Firing Scale",
        "desc": "Member count that triggers base firing frequency. More members = proportionally more firings if dynamic enabled.",
        "group": "variable_costs"
    },
    "MIN_FIRINGS_PER_MONTH": {
        "type": "int", "min": 1, "max": 15, "step": 1, "default": 4,
        "label": "Minimum Firings per Month",
        "desc": "Floor on monthly firings even with very few members. Ensures kiln maintenance and minimum service level.",
        "group": "variable_costs"
    },
    "MAX_FIRINGS_PER_MONTH": {
        "type": "int", "min": 8, "max": 50, "step": 1, "default": 12,
        "label": "Maximum Firings per Month",
        "desc": "Ceiling on monthly firings due to kiln capacity and staff time constraints. Prevents unrealistic firing schedules.",
        "group": "variable_costs"
    },
    
    # =============================================================================
    # OPERATIONAL COSTS: MAINTENANCE & MARKETING
    # =============================================================================
    "MAINTENANCE_BASE_COST": {
        "type": "float", "min": 50.0, "max": 1000.0, "step": 25.0, "default": 200.0,
        "label": "Base Monthly Maintenance ($)",
        "desc": "Predictable maintenance costs: kiln elements, wheel repairs, tool replacement. Core facility upkeep.",
        "group": "operational_costs"
    },
    "MAINTENANCE_RANDOM_STD": {
        "type": "float", "min": 0.0, "max": 500.0, "step": 25.0, "default": 150.0,
        "label": "Random Maintenance Variation ($)",
        "desc": "Standard deviation of unpredictable maintenance costs. Models equipment failures, emergency repairs, etc.",
        "group": "operational_costs"
    },
    "MARKETING_COST_BASE": {
        "type": "float", "min": 0.0, "max": 2000.0, "step": 50.0, "default": 300.0,
        "label": "Base Monthly Marketing ($)",
        "desc": "Ongoing marketing expenses: social media ads, materials, website. Essential for member acquisition.",
        "group": "operational_costs"
    },
    "MARKETING_RAMP_MONTHS": {
        "type": "int", "min": 1, "max": 24, "step": 1, "default": 12,
        "label": "Marketing Ramp Duration (months)",
        "desc": "Months of elevated marketing spending during startup. Higher spend needed to build initial awareness.",
        "group": "operational_costs"
    },
    "MARKETING_RAMP_MULTIPLIER": {
        "type": "float", "min": 1.0, "max": 5.0, "step": 0.25, "default": 2.0,
        "label": "Marketing Ramp Multiplier",
        "desc": "Marketing spend multiplier during ramp period. 2.0 = double spending for first 12 months to build awareness.",
        "group": "operational_costs"
    },
    
    # =============================================================================
    # STAFF COSTS
    # =============================================================================
    "STAFF_EXPANSION_THRESHOLD": {
        "type": "int", "min": 20, "max": 200, "step": 5, "default": 50,
        "label": "Staff Hiring Threshold (members)",
        "desc": "Member count that triggers hiring first employee. Represents owner capacity limits and service quality needs.",
        "group": "staff_costs"
    },
    "STAFF_COST_PER_MONTH": {
        "type": "float", "min": 1500.0, "max": 8000.0, "step": 100.0, "default": 2500.0,
        "label": "Monthly Staff Cost ($)",
        "desc": "Total monthly cost for first employee including wages, taxes, benefits. Part-time or full-time depending on needs.",
        "group": "staff_costs"
    },
    
    # =============================================================================
    # ENTITY TYPE & TAXATION
    # =============================================================================
    "ENTITY_TYPE": {
        "type": "select", "options": ["sole_prop", "partnership", "s_corp", "c_corp"], "default": "sole_prop",
        "label": "Business Entity Type",
        "desc": "Legal structure affecting taxation and owner compensation. Sole prop = simplest, S-corp = payroll taxes, C-corp = double taxation.",
        "group": "taxation"
    },
    "MA_PERSONAL_INCOME_TAX_RATE": {
        "type": "float", "min": 0.0, "max": 0.15, "step": 0.005, "default": 0.05,
        "label": "MA Personal Income Tax Rate",
        "desc": "Massachusetts personal income tax rate. Applies to pass-through entity income (sole prop, partnership, S-corp).",
        "group": "taxation"
    },
    "SE_SOC_SEC_RATE": {
        "type": "float", "min": 0.08, "max": 0.15, "step": 0.001, "default": 0.124,
        "label": "Self-Employment Social Security Rate",
        "desc": "Combined employer/employee Social Security rate for self-employed. Applies to SE income for sole prop/partnership.",
        "group": "taxation"
    },
    "SE_MEDICARE_RATE": {
        "type": "float", "min": 0.02, "max": 0.05, "step": 0.001, "default": 0.029,
        "label": "Self-Employment Medicare Rate",
        "desc": "Combined employer/employee Medicare rate for self-employed. Applies to all SE income with no wage base limit.",
        "group": "taxation"
    },
    "SE_SOC_SEC_WAGE_BASE": {
        "type": "int", "min": 100000, "max": 200000, "step": 1000, "default": 168600,
        "label": "SE Social Security Wage Base ($)",
        "desc": "Annual wage base limit for Social Security taxes. SE income above this is not subject to SS tax.",
        "group": "taxation"
    },
    "SCORP_OWNER_SALARY_PER_MONTH": {
        "type": "float", "min": 0.0, "max": 10000.0, "step": 100.0, "default": 4000.0,
        "label": "S-Corp Owner Monthly Salary ($)",
        "desc": "Required reasonable salary for S-corp owner. Subject to payroll taxes but avoids SE tax on profits above salary.",
        "group": "taxation"
    },
    "FED_CORP_TAX_RATE": {
        "type": "float", "min": 0.15, "max": 0.35, "step": 0.01, "default": 0.21,
        "label": "Federal Corporate Tax Rate",
        "desc": "Federal income tax rate for C-corporations. Applied to corporate profits before dividends to owners.",
        "group": "taxation"
    },
    "MA_CORP_TAX_RATE": {
        "type": "float", "min": 0.05, "max": 0.12, "step": 0.005, "default": 0.08,
        "label": "MA Corporate Tax Rate",
        "desc": "Massachusetts corporate income tax rate. Combined with federal rate for total C-corp tax burden.",
        "group": "taxation"
    },
    "MA_SALES_TAX_RATE": {
        "type": "float", "min": 0.0, "max": 0.15, "step": 0.005, "default": 0.0625,
        "label": "MA Sales Tax Rate",
        "desc": "Massachusetts sales tax rate applied to clay sales. Must be collected and remitted quarterly.",
        "group": "taxation"
    },
    
    # =============================================================================
    # FINANCING: SBA LOANS
    # =============================================================================
    # SBA Loan Amounts (Auto-calculated from CapEx and OpEx)
    "LOAN_504_AMOUNT_OVERRIDE": {
        "type": "float", "min": 0.0, "max": 500000.0, "step": 1000.0, "default": 0.0,
        "label": "SBA 504 Loan Amount Override ($, 0=Auto-calculate)",
        "desc": "Manual override for SBA 504 loan amount. Leave at 0 to auto-calculate from CapEx equipment costs plus contingency.",
        "group": "financing"
    },
    "LOAN_7A_AMOUNT_OVERRIDE": {
        "type": "float", "min": 0.0, "max": 500000.0, "step": 1000.0, "default": 0.0,
        "label": "SBA 7(a) Loan Amount Override ($, 0=Auto-calculate)", 
        "desc": "Manual override for SBA 7(a) loan amount. Leave at 0 to auto-calculate from 8 months of OpEx (rent + owner draw + insurance).",
        "group": "financing"
    },
    "LOAN_504_ANNUAL_RATE": {
        "type": "float", "min": 0.03, "max": 0.15, "step": 0.001, "default": 0.070,
        "label": "SBA 504 Annual Rate",
        "desc": "Blended interest rate for SBA 504 loan (equipment/real estate). Typically lower than conventional financing.",
        "group": "financing"
    },
    "LOAN_504_TERM_YEARS": {
        "type": "int", "min": 5, "max": 25, "step": 1, "default": 20,
        "label": "SBA 504 Term (years)",
        "desc": "Repayment period for 504 loan. Longer terms available for real estate (20 years) vs equipment (10-15 years).",
        "group": "financing"
    },
    "IO_MONTHS_504": {
        "type": "int", "min": 0, "max": 18, "step": 1, "default": 6,
        "label": "504 Interest-Only Months",
        "desc": "Initial months with interest-only payments on 504 loan. Helps cash flow during startup phase.",
        "group": "financing"
    },
    "LOAN_7A_ANNUAL_RATE": {
        "type": "float", "min": 0.05, "max": 0.20, "step": 0.001, "default": 0.115,
        "label": "SBA 7(a) Annual Rate",
        "desc": "Interest rate for SBA 7(a) loan (working capital/general business). Higher than 504 but more flexible use.",
        "group": "financing"
    },
    "LOAN_7A_TERM_YEARS": {
        "type": "int", "min": 5, "max": 10, "step": 1, "default": 7,
        "label": "SBA 7(a) Term (years)",
        "desc": "Repayment period for 7(a) loan. Typically shorter than 504, reflecting working capital vs fixed asset nature.",
        "group": "financing"
    },
    "IO_MONTHS_7A": {
        "type": "int", "min": 0, "max": 18, "step": 1, "default": 6,
        "label": "7(a) Interest-Only Months",
        "desc": "Initial months with interest-only payments on 7(a) loan. Preserves working capital during startup.",
        "group": "financing"
    },
    "LOAN_CONTINGENCY_PCT": {
        "type": "float", "min": 0.0, "max": 0.30, "step": 0.01, "default": 0.08,
        "label": "CapEx Contingency %",
        "desc": "Percentage buffer added to equipment costs for loan sizing. Accounts for cost overruns and unexpected expenses.",
        "group": "financing"
    },
    "RUNWAY_MONTHS": {
        "type": "int", "min": 6, "max": 24, "step": 1, "default": 12,
        "label": "Operating Runway (months)",
        "desc": "Months of operating expenses to include in 7(a) loan sizing. Higher = more cushion but more debt service.",
        "group": "financing"
    },
    "EXTRA_BUFFER": {
        "type": "float", "min": 0.0, "max": 50000.0, "step": 1000.0, "default": 10000.0,
        "label": "Extra Working Capital Buffer ($)",
        "desc": "Additional cash buffer beyond calculated runway. Conservative approach for uncertain markets or complex operations.",
        "group": "financing"
    },
    "RESERVE_FLOOR": {
        "type": "float", "min": 0.0, "max": 50000.0, "step": 1000.0, "default": 5000.0,
        "label": "Minimum Cash Reserve ($)",
        "desc": "Minimum cash balance to maintain. Used for line of credit sizing and cash management policies.",
        "group": "financing"
    },
    
    # SBA Fees
    "FEES_UPFRONT_PCT_7A": {
        "type": "float", "min": 0.0, "max": 0.05, "step": 0.0025, "default": 0.03,
        "label": "7(a) Upfront Fee %",
        "desc": "SBA guarantee fee as percentage of 7(a) loan amount. Typically 2-3.5% depending on loan size.",
        "group": "financing"
    },
    "FEES_UPFRONT_PCT_504": {
        "type": "float", "min": 0.0, "max": 0.05, "step": 0.0025, "default": 0.02,
        "label": "504 Upfront Fee %",
        "desc": "SBA guarantee fee as percentage of 504 loan amount. Generally lower than 7(a) fees.",
        "group": "financing"
    },
    "FEES_PACKAGING": {
        "type": "float", "min": 0.0, "max": 10000.0, "step": 250.0, "default": 2500.0,
        "label": "Loan Packaging Fee ($)",
        "desc": "Professional fees for loan application preparation. Paid to consultants or packagers who prepare SBA applications.",
        "group": "financing"
    },
    "FEES_CLOSING": {
        "type": "float", "min": 0.0, "max": 5000.0, "step": 100.0, "default": 1500.0,
        "label": "Loan Closing Costs ($)",
        "desc": "Legal, title, and closing costs for loan finalization. One-time expense at loan funding.",
        "group": "financing"
    },
    "FINANCE_FEES_7A": {
        "type": "bool", "default": True,
        "label": "Finance 7(a) Fees into Loan",
        "desc": "Whether to roll 7(a) fees into loan principal vs pay cash. Financing preserves cash but increases debt service.",
        "group": "financing"
    },
    "FINANCE_FEES_504": {
        "type": "bool", "default": True,
        "label": "Finance 504 Fees into Loan",
        "desc": "Whether to roll 504 fees into loan principal vs pay cash. Financing preserves cash but increases debt service.",
        "group": "financing"
    },
    
    # =============================================================================
    # GRANTS & EXTERNAL FUNDING
    # =============================================================================
    "grant_amount": {
        "type": "float", "min": 0.0, "max": 100000.0, "step": 1000.0, "default": 0.0,
        "label": "Grant Amount ($)",
        "desc": "One-time grant funding amount. Can model CDBG, arts grants, COVID relief, or other non-repayable funding.",
        "group": "grants"
    },
    "grant_month": {
        "type": "int", "min": -1, "max": 60, "step": 1, "default": -1,
        "label": "Grant Timing (month, -1=None)",
        "desc": "Month when grant funds are received (1-based). Set to -1 for no grant. Timing affects cash flow and survival probability.",
        "group": "grants"
    },
    
    # =============================================================================
    # MEMBERSHIP TRAJECTORY MODE
    # =============================================================================
    "MEMBERSHIP_MODE": {
        "type": "select", 
        "options": ["calculated", "manual_table", "piecewise_trends"], 
        "default": "calculated",
        "label": "Membership Projection Method",
        "desc": "How to determine membership over time: calculated from market dynamics, manual month-by-month input, or piecewise trend specification.",
        "group": "membership_trajectory"
    },
    "MONTHS": {
        "type": "int", "min": 12, "max": 120, "step": 6, "default": 60,
        "label": "Simulation Horizon (months)",
        "desc": "Total months to simulate. Longer horizons show mature operations but increase runtime. 60 months = 5 years typical.",
        "group": "simulation"
    },
    "N_SIMULATIONS": {
        "type": "int", "min": 10, "max": 1000, "step": 10, "default": 100,
        "label": "Number of Simulations",
        "desc": "Monte Carlo simulations to run. More = better statistics but longer runtime. 100+ recommended for reliable percentiles.",
        "group": "simulation"
    },
    "RANDOM_SEED": {
        "type": "int", "min": 1, "max": 999999, "step": 1, "default": 42,
        "label": "Random Seed",
        "desc": "Random number generator seed for reproducible results. Change to get different random scenarios with same parameters.",
        "group": "simulation"
    },
}

# --- Ensure separate 504 buffer parameter exists (for misc CapEx not captured elsewhere)
if "EXTRA_504_BUFFER" not in COMPLETE_PARAM_SPECS:
    COMPLETE_PARAM_SPECS["EXTRA_504_BUFFER"] = {
        "type": "float", "min": 0.0, "max": 200000.0, "step": 500.0, "default": 0.0,
        "label": "SBA 504 Misc Buffer ($)",
        "desc": "Extra amount to add on top of 504-eligible equipment total + contingency. Use for buildout odds-and-ends not itemized.",
        "group": "financing"
    }


# GROUP DEFINITIONS WITH LOGICAL ORGANIZATION
PARAMETER_GROUPS = {
    "membership_trajectory": {
        "title": "📈 Membership Trajectory", 
        "color": "green",
        "desc": "How membership grows over time - the most critical business assumption.",
        "priority": 1
    },
    "business_fundamentals": {
        "title": "🏢 Business Fundamentals", 
        "color": "green",
        "desc": "Core business parameters most likely to vary between studios and locations.",
        "priority": 2
    },
    "pricing": {
        "title": "💰 Pricing & Market Response", 
        "color": "amber", 
        "desc": "Pricing strategy and how customers respond to price changes.",
        "priority": 3
    },
    "member_behavior": {
        "title": "👥 Member Behavior & Archetypes", 
        "color": "amber",
        "desc": "Member mix, usage patterns, and retention characteristics by member type.",
        "priority": 4
    },
    "capacity": {
        "title": "🏭 Studio Capacity & Equipment", 
        "color": "green",
        "desc": "Physical capacity constraints, equipment counts, and utilization factors.",
        "priority": 5
    },
    "market_dynamics": {
        "title": "📈 Market Dynamics & Acquisition", 
        "color": "amber",
        "desc": "Market size, acquisition channels, word-of-mouth, and member acquisition rates. Only used if Membership Mode = 'calculated'.",
        "priority": 6
    },
    "economic_environment": {
        "title": "🌊 Economic Environment", 
        "color": "red",
        "desc": "Economic stress frequency and impact on member behavior.",
        "priority": 6
    },
    "seasonality": {
        "title": "🗓️ Seasonal Patterns", 
        "color": "amber",
        "desc": "Monthly activity multipliers reflecting seasonal demand patterns.",
        "priority": 7
    },
    "clay_firing_revenue": {
        "title": "🏺 Clay & Firing Revenue", 
        "color": "amber",
        "desc": "Clay sales pricing, costs, and member usage patterns by archetype.",
        "priority": 8
    },
    "workshops": {
        "title": "🎨 Workshop Revenue Stream", 
        "color": "amber",
        "desc": "Short pottery experiences for beginners - pricing, costs, and conversion rates.",
        "priority": 9
    },
    "classes": {
        "title": "🎓 Class Revenue Stream", 
        "color": "amber", 
        "desc": "Multi-week structured courses - scheduling, pricing, and member conversion.",
        "priority": 10
    },
    "events": {
        "title": "🎉 Event Revenue Stream", 
        "color": "amber",
        "desc": "Public events like paint-a-pot parties - frequency, pricing, and costs.",
        "priority": 11
    },
    "designated_studios": {
        "title": "🏠 Designated Studio Rentals", 
        "color": "amber",
        "desc": "Premium private workspace rentals for serious artists.",
        "priority": 12
    },
    "fixed_costs": {
        "title": "🏢 Fixed Operating Costs", 
        "color": "green",
        "desc": "Predictable monthly expenses - insurance, utilities, supplies.",
        "priority": 13
    },
    "variable_costs": {
        "title": "⚡ Variable Operating Costs", 
        "color": "amber",
        "desc": "Usage-based costs - electricity, water, kiln operations.",
        "priority": 14
    },
    "operational_costs": {
        "title": "🔧 Operational Costs", 
        "color": "amber",
        "desc": "Maintenance, marketing, and other operational expenses.",
        "priority": 15
    },
    "staff_costs": {
        "title": "👨‍💼 Staffing Costs", 
        "color": "amber",
        "desc": "Employee costs and hiring triggers based on member growth.",
        "priority": 16
    },
    "taxation": {
        "title": "💸 Taxation & Entity Structure", 
        "color": "red",
        "desc": "Business entity type and tax rates - rarely changed during planning.",
        "priority": 17
    },
    "financing": {
        "title": "🏦 SBA Loan Financing", 
        "color": "red",
        "desc": "Loan terms, rates, fees, and sizing parameters.",
        "priority": 18
    },
    "grants": {
        "title": "🎁 Grants & External Funding", 
        "color": "red",
        "desc": "Non-repayable funding sources and timing.",
        "priority": 19
    },
    "operations": {
        "title": "⚙️ Operational Constraints", 
        "color": "amber",
        "desc": "Operational limits and capacity constraints.",
        "priority": 20
    },
    "simulation": {
        "title": "🎯 Simulation Settings", 
        "color": "blue",
        "desc": "Monte Carlo simulation parameters - horizon, iterations, randomness.",
        "priority": 21
    }
}

def calculate_loan_metrics(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate comprehensive loan analysis metrics from simulation results"""
    
    # Extract loan parameters
    loan_504_rate = params.get("LOAN_504_ANNUAL_RATE", 0.070)
    loan_7a_rate = params.get("LOAN_7A_ANNUAL_RATE", 0.115)
    loan_504_term = params.get("LOAN_504_TERM_YEARS", 20)
    loan_7a_term = params.get("LOAN_7A_TERM_YEARS", 7)
    io_months_504 = params.get("IO_MONTHS_504", 6)
    io_months_7a = params.get("IO_MONTHS_7A", 6)
    
    # Calculate 504 amount from CapEx items (use user override if provided)
    if "LOAN_504_AMOUNT_OVERRIDE" in params and params["LOAN_504_AMOUNT_OVERRIDE"] > 0:
        total_504_amount = params["LOAN_504_AMOUNT_OVERRIDE"]
    else:
        # Robustly coerce & filter CAPEX rows before summing
        raw_items = params.get("CAPEX_ITEMS", []) or []
        try:
            capex_df = pd.DataFrame(raw_items)
        except Exception:
            capex_df = pd.DataFrame([])

        if not capex_df.empty:
            # Ensure required columns exist
            for col in ("enabled", "finance_504", "unit_cost", "count"):
                if col not in capex_df.columns:
                    capex_df[col] = [True] * len(capex_df) if col in ("enabled", "finance_504") else 0

            # Type coercion (prevents "string x int" oddities)
            capex_df["unit_cost"] = pd.to_numeric(capex_df["unit_cost"], errors="coerce").fillna(0.0)
            capex_df["count"]     = pd.to_numeric(capex_df["count"],     errors="coerce").fillna(1).astype(int)
            capex_df["enabled"]   = capex_df["enabled"].fillna(True).astype(bool)
            capex_df["finance_504"] = capex_df["finance_504"].fillna(True).astype(bool)

            sel = capex_df["enabled"] & capex_df["finance_504"]
            total_504_amount = float((capex_df.loc[sel, "unit_cost"] * capex_df.loc[sel, "count"]).sum())
        else:
            total_504_amount = 0.0

        # Apply contingency & explicit buffer
        contingency = float(params.get("LOAN_CONTINGENCY_PCT", 0.08) or 0.0)
        total_504_amount *= (1.0 + contingency)
        total_504_amount += float(params.get("EXTRA_504_BUFFER", 0.0) or 0.0)
    
    # Calculate 7(a) amount from OpEx (use user override if provided)
    if "LOAN_7A_AMOUNT_OVERRIDE" in params and params["LOAN_7A_AMOUNT_OVERRIDE"] > 0:
        total_7a_amount = params["LOAN_7A_AMOUNT_OVERRIDE"]
    else:
        # Base OpEx calculation: rent + owner draw + basic fixed costs
        monthly_rent = params.get("RENT", 3500)
        monthly_owner_draw = params.get("OWNER_DRAW", 2000)
        monthly_insurance = params.get("INSURANCE_COST", 75)
        monthly_base_opex = monthly_rent + monthly_owner_draw + monthly_insurance
        
        # Use RUNWAY_MONTHS when provided for 7(a) sizing
        runway_months = int(params.get("RUNWAY_MONTHS", 8) or 8)
        extra_buffer = params.get("EXTRA_BUFFER", 10000)
        total_7a_amount = (monthly_base_opex * runway_months) + extra_buffer
    
    # Calculate monthly payments
    def calculate_monthly_payment(principal, annual_rate, term_years, io_months=0):
        """Calculate monthly payment for loan with optional interest-only period"""
        monthly_rate = annual_rate / 12
        if monthly_rate == 0:
            return principal / (term_years * 12 - io_months)
        
        # Interest-only payment during IO period
        io_payment = principal * monthly_rate if io_months > 0 else 0
        
        # Amortizing payment after IO period
        amort_months = term_years * 12 - io_months
        if amort_months <= 0:
            return principal / (term_years * 12)
            
        amort_payment = principal * (monthly_rate * (1 + monthly_rate)**amort_months) / ((1 + monthly_rate)**amort_months - 1)
        
        return io_payment, amort_payment
    
    # Calculate payments for both loans
    payments_504 = calculate_monthly_payment(total_504_amount, loan_504_rate, loan_504_term, io_months_504)
    payments_7a = calculate_monthly_payment(total_7a_amount, loan_7a_rate, loan_7a_term, io_months_7a)
    
    if isinstance(payments_504, tuple):
        io_payment_504, amort_payment_504 = payments_504
    else:
        io_payment_504, amort_payment_504 = 0, payments_504
        
    if isinstance(payments_7a, tuple):
        io_payment_7a, amort_payment_7a = payments_7a
    else:
        io_payment_7a, amort_payment_7a = 0, payments_7a
    
    # Calculate outstanding balances over time
    months = len(df[df["simulation_id"] == df["simulation_id"].iloc[0]])
    outstanding_504 = []
    outstanding_7a = []
    monthly_payments_504 = []
    monthly_payments_7a = []
    
    balance_504 = total_504_amount
    balance_7a = total_7a_amount
    
    for month in range(1, months + 1):
        # 504 loan
        if month <= io_months_504:
            payment_504 = io_payment_504
            principal_payment_504 = 0
        else:
            payment_504 = amort_payment_504
            principal_payment_504 = payment_504 - (balance_504 * loan_504_rate / 12)
            balance_504 = max(0, balance_504 - principal_payment_504)
        
        # 7(a) loan
        if month <= io_months_7a:
            payment_7a = io_payment_7a
            principal_payment_7a = 0
        else:
            payment_7a = amort_payment_7a
            principal_payment_7a = payment_7a - (balance_7a * loan_7a_rate / 12)
            balance_7a = max(0, balance_7a - principal_payment_7a)
        
        outstanding_504.append(balance_504)
        outstanding_7a.append(balance_7a)
        monthly_payments_504.append(payment_504)
        monthly_payments_7a.append(payment_7a)
    
    return {
        "total_504_amount": total_504_amount,
        "total_7a_amount": total_7a_amount,
        "io_payment_504": io_payment_504,
        "amort_payment_504": amort_payment_504,
        "io_payment_7a": io_payment_7a,
        "amort_payment_7a": amort_payment_7a,
        "outstanding_504": outstanding_504,
        "outstanding_7a": outstanding_7a,
        "monthly_payments_504": monthly_payments_504,
        "monthly_payments_7a": monthly_payments_7a,
    }

def calculate_dscr_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate DSCR analysis metrics from simulation results"""
    
    # Find DSCR column
    dscr_col = pick_col(df, ["dscr", "debt_service_coverage_ratio", "DSCR"])
    
    if not dscr_col:
        return {"error": "No DSCR data found in simulation results"}
    
    # Clean DSCR data - replace infinite values
    dscr_data = df[dscr_col].replace([np.inf, -np.inf], np.nan).dropna()
    
    if dscr_data.empty:
        return {"error": "No valid DSCR data available"}
    
    # Multi-timepoint analysis (Years 1, 2, 3, 5)
    timepoint_analysis = {}
    for year in [1, 2, 3, 5]:
        month = year * 12
        year_data = df[df["month"] == month][dscr_col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if not year_data.empty:
            timepoint_analysis[f"year_{year}"] = {
                "mean": year_data.mean(),
                "median": year_data.median(),
                "p10": year_data.quantile(0.1),
                "p25": year_data.quantile(0.25),
                "p75": year_data.quantile(0.75),
                "p90": year_data.quantile(0.9),
                "below_125": (year_data < 1.25).mean(),
                "below_100": (year_data < 1.0).mean(),
                "count": len(year_data)
            }
    
    # Risk assessment - percentage below critical thresholds
    risk_assessment = {
        "below_125_pct": (dscr_data < 1.25).mean() * 100,
        "below_100_pct": (dscr_data < 1.0).mean() * 100,
        "mean_dscr": dscr_data.mean(),
        "median_dscr": dscr_data.median(),
        "std_dscr": dscr_data.std(),
        "min_dscr": dscr_data.min(),
        "max_dscr": dscr_data.max()
    }
    
    # DSCR evolution by month (for trending)
    dscr_evolution = df.groupby("month")[dscr_col].agg([
        "count", "mean", "median", "std", 
        lambda x: x.replace([np.inf, -np.inf], np.nan).quantile(0.1),
        lambda x: x.replace([np.inf, -np.inf], np.nan).quantile(0.9)
    ]).reset_index()
    
    dscr_evolution.columns = ["month", "count", "mean", "median", "std", "p10", "p90"]
    
    return {
        "timepoint_analysis": timepoint_analysis,
        "risk_assessment": risk_assessment, 
        "dscr_evolution": dscr_evolution,
        "dscr_col": dscr_col
    }

def render_loan_analysis(df: pd.DataFrame, params_state: Dict[str, Any]):
    """Render comprehensive loan analysis section"""
    
    st.header("🏦 Loan Analysis & DSCR Assessment")
    st.markdown("Professional loan analysis with debt service coverage ratio (DSCR) metrics that SBA lenders expect.")
    
    # Calculate loan metrics (suggested principals & payments)
    loan_metrics = calculate_loan_metrics(df, params_state)
    calc_504 = int(loan_metrics.get("total_504_amount", 0) or 0)
    calc_7a  = int(loan_metrics.get("total_7a_amount", 0) or 0)
    # Effective principals used for payments/DSCR: override if >0 else suggestion
    override_504 = int(params_state.get("LOAN_504_AMOUNT_OVERRIDE", 0) or 0)
    override_7a  = int(params_state.get("LOAN_7A_AMOUNT_OVERRIDE", 0) or 0)
    used_504 = override_504 if override_504 > 0 else calc_504
    used_7a  = override_7a  if override_7a  > 0 else calc_7a
    
    # Enhanced Loan Amount Displays (always rendered)
    with st.expander("💰 Loan Amount Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SBA 504 Loan (CapEx)")
            st.metric("Used Loan Amount (Override or Suggested)", f"${used_504:,.0f}")
            st.caption(f"Suggested 504 principal: ${calc_504:,.0f}"
                       + (" • Override active" if override_504 > 0 else " • Auto-calc"))

            st.metric("Interest-Only Payment", f"${loan_metrics['io_payment_504']:,.0f}/month")
            st.metric("Amortizing Payment", f"${loan_metrics['amort_payment_504']:,.0f}/month")
 
        with col2:
            st.subheader("SBA 7(a) Loan (OpEx)")
            st.metric("Used Loan Amount (Override or Suggested)", f"${used_7a:,.0f}")
            st.caption(f"Suggested 7(a) principal: ${calc_7a:,.0f}"
                       + (" • Override active" if override_7a > 0 else " • Auto-calc"))

            st.metric("Interest-Only Payment", f"${loan_metrics['io_payment_7a']:,.0f}/month")
            st.metric("Amortizing Payment", f"${loan_metrics['amort_payment_7a']:,.0f}/month")

        
        # Total debt service
        st.subheader("Combined Debt Service")
        total_io = loan_metrics['io_payment_504'] + loan_metrics['io_payment_7a']
        total_amort = loan_metrics['amort_payment_504'] + loan_metrics['amort_payment_7a']
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Debt", f"${used_504 + used_7a:,.0f}")
        col2.metric("IO Period Payment", f"${total_io:,.0f}/month")
        col3.metric("Amortizing Payment", f"${total_amort:,.0f}/month")

    # DSCR and charts can fail without hiding the expander; compute DSCR after
    try:
        dscr_metrics = calculate_dscr_metrics(df)
    except Exception as _e:
        dscr_metrics = {"error": str(_e)}
    
    # Loan Repayment Charts
    with st.expander("📊 Loan Repayment Visualization", expanded=False):
        months_range = list(range(1, len(loan_metrics["outstanding_504"]) + 1))
        
        # Dual panel chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Outstanding balances
        ax1.plot(months_range, loan_metrics["outstanding_504"], label="504 Loan", linewidth=2, color='#1f77b4')
        ax1.plot(months_range, loan_metrics["outstanding_7a"], label="7(a) Loan", linewidth=2, color='#ff7f0e')
        ax1.set_title("Outstanding Loan Balances")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Outstanding Balance ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Monthly payments
        ax2.plot(months_range, loan_metrics["monthly_payments_504"], label="504 Payment", linewidth=2, color='#1f77b4')
        ax2.plot(months_range, loan_metrics["monthly_payments_7a"], label="7(a) Payment", linewidth=2, color='#ff7f0e')
        total_payments = [p504 + p7a for p504, p7a in zip(loan_metrics["monthly_payments_504"], loan_metrics["monthly_payments_7a"])]
        ax2.plot(months_range, total_payments, label="Total Payment", linewidth=3, color='#d62728', linestyle='--')
        ax2.set_title("Monthly Debt Service Payments")
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Monthly Payment ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # DSCR Analysis
    if "error" not in dscr_metrics:
        # Multi-Timepoint DSCR Analysis
        with st.expander("📈 DSCR Analysis by Time Period", expanded=True):
            timepoint_data = dscr_metrics["timepoint_analysis"]
            
            if timepoint_data:
                # Create metrics display
                years = [1, 2, 3, 5]
                available_years = [y for y in years if f"year_{y}" in timepoint_data]
                
                cols = st.columns(len(available_years))
                
                for i, year in enumerate(available_years):
                    data = timepoint_data[f"year_{year}"]
                    
                    with cols[i]:
                        st.subheader(f"Year {year}")
                        
                        # Color-coded risk indicators
                        median_dscr = data["median"]
                        if median_dscr >= 1.25:
                            color = "green"
                            risk_level = "Low Risk"
                        elif median_dscr >= 1.0:
                            color = "orange" 
                            risk_level = "Moderate Risk"
                        else:
                            color = "red"
                            risk_level = "High Risk"
                        
                        st.metric("Median DSCR", f"{median_dscr:.2f}", help=f"Risk Level: {risk_level}")
                        st.metric("10th Percentile", f"{data['p10']:.2f}")
                        st.metric("90th Percentile", f"{data['p90']:.2f}")
                        st.metric("Below 1.25x", f"{data['below_125']*100:.1f}%")
                        st.metric("Below 1.0x", f"{data['below_100']*100:.1f}%")
        
        # DSCR Evolution Chart
        with st.expander("📊 DSCR Trend Analysis", expanded=False):
            dscr_evolution = dscr_metrics["dscr_evolution"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Main trend line
            ax.plot(dscr_evolution["month"], dscr_evolution["median"], 
                   linewidth=3, color='#1f77b4', label='Median DSCR')
            
            # Confidence bands
            ax.fill_between(dscr_evolution["month"], 
                           dscr_evolution["p10"], 
                           dscr_evolution["p90"],
                           alpha=0.3, color='#1f77b4', label='10th-90th Percentile')
            
            # Reference lines
            ax.axhline(y=1.25, color='orange', linestyle='--', alpha=0.7, label='1.25x Threshold (Preferred)')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1.0x Threshold (Minimum)')
            
            ax.set_title("DSCR Evolution Over Time")
            ax.set_xlabel("Month")
            ax.set_ylabel("Debt Service Coverage Ratio")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, min(5.0, dscr_evolution["p90"].max() * 1.1))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # DSCR Risk Assessment
        with st.expander("⚠️ DSCR Risk Assessment", expanded=True):
            risk_data = dscr_metrics["risk_assessment"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Overall DSCR Statistics")
                st.metric("Mean DSCR", f"{risk_data['mean_dscr']:.2f}")
                st.metric("Median DSCR", f"{risk_data['median_dscr']:.2f}")
                st.metric("Standard Deviation", f"{risk_data['std_dscr']:.2f}")
            
            with col2:
                st.subheader("Risk Thresholds")
                below_125 = risk_data['below_125_pct']
                below_100 = risk_data['below_100_pct']
                
                # Color-coded risk metrics
                if below_125 < 10:
                    risk_color_125 = "green"
                elif below_125 < 25:
                    risk_color_125 = "orange"
                else:
                    risk_color_125 = "red"
                
                if below_100 < 5:
                    risk_color_100 = "green"
                elif below_100 < 15:
                    risk_color_100 = "orange"
                else:
                    risk_color_100 = "red"
                
                st.metric("Below 1.25x", f"{below_125:.1f}%", 
                         delta=f"Risk: {'Low' if below_125 < 10 else 'Moderate' if below_125 < 25 else 'High'}")
                st.metric("Below 1.0x", f"{below_100:.1f}%",
                         delta=f"Risk: {'Low' if below_100 < 5 else 'Moderate' if below_100 < 15 else 'High'}")
            
            with col3:
                st.subheader("Range")
                st.metric("Minimum DSCR", f"{risk_data['min_dscr']:.2f}")
                st.metric("Maximum DSCR", f"{risk_data['max_dscr']:.2f}")
        
        # Enhanced Matrix Heatmaps
        with st.expander("📊 DSCR Risk Matrix", expanded=False):
            st.subheader("DSCR Risk Assessment Matrix")
            
            # Create DSCR risk matrix by member count and month
            if "active_members" in df.columns:
                # Bin members into ranges
                df_clean = df.copy()
                df_clean["dscr_clean"] = df_clean[dscr_metrics["dscr_col"]].replace([np.inf, -np.inf], np.nan)
                df_clean = df_clean.dropna(subset=["dscr_clean"])
                
                if len(df_clean) > 0:
                    # Create member bins based on actual data distribution
                    max_members = df_clean['active_members'].max()
                    if max_members <= 50:
                        member_bins = [0, 15, 30, 45, float('inf')]
                        member_labels = ['0-15', '16-30', '31-45', '46+']
                    else:
                        member_bins = [0, 25, 50, 75, float('inf')]
                        member_labels = ['0-25', '26-50', '51-75', '76+']
                    
                    df_clean['member_bin'] = pd.cut(df_clean['active_members'], 
                                                  bins=member_bins, labels=member_labels, right=True)
                    
                    # Create time bins (years only for cleaner display)
                    df_clean['year'] = ((df_clean['month'] - 1) // 12) + 1
                    df_clean = df_clean[df_clean['year'] <= 5]  # Limit to first 5 years
                    df_clean['time_period'] = 'Year ' + df_clean['year'].astype(str)
                    
                    # Calculate risk metrics for matrix
                    risk_matrix = df_clean.groupby(['member_bin', 'time_period']).agg({
                        'dscr_clean': ['mean', 'count', lambda x: (x < 1.25).mean() * 100]
                    }).reset_index()
                    
                    risk_matrix.columns = ['member_bin', 'time_period', 'mean_dscr', 'count', 'risk_pct']
                    
                    # Only show matrix if we have sufficient data
                    if len(risk_matrix) > 0 and risk_matrix['count'].sum() >= 20:
                        # Pivot for heatmap
                        heatmap_data = risk_matrix.pivot(index='member_bin', columns='time_period', values='mean_dscr')
                        risk_heatmap_data = risk_matrix.pivot(index='member_bin', columns='time_period', values='risk_pct')
                        
                        # Create cleaner, larger heatmaps
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # Mean DSCR heatmap
                        if not heatmap_data.empty:
                            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                                       center=1.25, ax=ax1, cbar_kws={'label': 'Mean DSCR'},
                                       square=False, linewidths=0.5)
                            ax1.set_title('Mean DSCR by Member Count and Time Period', fontsize=14, pad=20)
                            ax1.set_xlabel('Time Period', fontsize=12)
                            ax1.set_ylabel('Member Count Range', fontsize=12)
                            ax1.tick_params(axis='x', rotation=0)
                            ax1.tick_params(axis='y', rotation=0)
                        
                        # Risk percentage heatmap
                        if not risk_heatmap_data.empty:
                            sns.heatmap(risk_heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                                       ax=ax2, cbar_kws={'label': '% Below 1.25x DSCR'},
                                       square=False, linewidths=0.5)
                            ax2.set_title('DSCR Risk Percentage by Member Count and Time Period', fontsize=14, pad=20)
                            ax2.set_xlabel('Time Period', fontsize=12)
                            ax2.set_ylabel('Member Count Range', fontsize=12)
                            ax2.tick_params(axis='x', rotation=0)
                            ax2.tick_params(axis='y', rotation=0)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add interpretation guide
                        st.markdown("**Matrix Interpretation:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("- **Green areas**: Strong DSCR (>1.25x), low risk")
                            st.markdown("- **Yellow areas**: Moderate DSCR (1.0-1.25x), moderate risk")
                        with col2:
                            st.markdown("- **Red areas**: Weak DSCR (<1.0x), high risk")
                            st.markdown("- **Blue areas**: Low risk percentage (<20%)")
                    else:
                        st.info("Insufficient data points for reliable matrix analysis. Need more simulation results or longer time horizon.")
                else:
                    st.info("No valid DSCR data available for matrix analysis")
            else:
                st.info("Member count data not available for matrix analysis")
    
    else:
        st.warning(f"DSCR Analysis unavailable: {dscr_metrics['error']}")
        st.info("DSCR data may not be generated by your current simulation configuration.")

# PARAMETER RENDERING FUNCTIONS

def render_membership_trajectory(params_state: dict) -> dict:
    """Special rendering for membership trajectory options"""
    
    membership_mode = params_state.get("MEMBERSHIP_MODE", "calculated")
    
    st.markdown("**📈 Membership Trajectory**")
    st.markdown("🟢 The most critical business assumption - how membership grows over time.")
    
    # Mode selector
    params_state["MEMBERSHIP_MODE"] = st.selectbox(
        "Membership Projection Method",
        options=["calculated", "manual_table", "piecewise_trends"],
        index=["calculated", "manual_table", "piecewise_trends"].index(membership_mode),
        format_func=lambda x: {
            "calculated": "Calculated from Market Dynamics (original model)",
            "manual_table": "Manual Table (specify exact member count each month)",
            "piecewise_trends": "Piecewise Trends (specify growth patterns by period)"
        }[x],
        help="Choose how to determine membership over time. Manual options let you specify your own growth assumptions."
    )
    
    if params_state["MEMBERSHIP_MODE"] == "manual_table":
        st.markdown("**Manual Membership Table**")
        st.caption("Specify exact member count for each month. Model will use these values instead of calculating member acquisition.")
        
        months = params_state.get("MONTHS", 60)
        
        # Initialize with reasonable defaults if not exists
        if "MANUAL_MEMBERSHIP_TABLE" not in params_state:
            # Create a reasonable growth curve as default
            default_curve = []
            for month in range(1, months + 1):
                if month <= 6:
                    members = min(20, month * 3)  # Start slow
                elif month <= 24:
                    members = 20 + (month - 6) * 1.5  # Steady growth
                else:
                    members = min(60, 47 + (month - 24) * 0.5)  # Slower mature growth
                default_curve.append(int(members))
            params_state["MANUAL_MEMBERSHIP_TABLE"] = default_curve
        
        # Create DataFrame for editing
        membership_df = pd.DataFrame({
            "Month": list(range(1, months + 1)),
            "Members": params_state["MANUAL_MEMBERSHIP_TABLE"][:months]
        })
        
        # Data editor
        edited_df = st.data_editor(
            membership_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Month": st.column_config.NumberColumn("Month", disabled=True),
                "Members": st.column_config.NumberColumn("Members", min_value=0, max_value=500, step=1)
            },
            hide_index=True
        )
        
        params_state["MANUAL_MEMBERSHIP_TABLE"] = edited_df["Members"].tolist()
        
        # Show a preview chart
        if len(edited_df) > 0:
            st.line_chart(edited_df.set_index("Month")["Members"])
    
    elif params_state["MEMBERSHIP_MODE"] == "piecewise_trends":
        st.markdown("**Piecewise Trend Specification**")
        st.caption("Define growth patterns for different time periods. Each segment can be linear or exponential.")
        
        # Initialize with default segments
        if "MEMBERSHIP_SEGMENTS" not in params_state:
            params_state["MEMBERSHIP_SEGMENTS"] = [
                {"start_month": 1, "end_month": 6, "start_members": 5, "end_members": 20, "type": "linear"},
                {"start_month": 7, "end_month": 24, "start_members": 20, "end_members": 50, "type": "linear"},
                {"start_month": 25, "end_month": 60, "start_members": 50, "end_members": 70, "type": "exponential"}
            ]
        
        # Segment editor
        segments_df = pd.DataFrame(params_state["MEMBERSHIP_SEGMENTS"])
        
        edited_segments = st.data_editor(
            segments_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "start_month": st.column_config.NumberColumn("Start Month", min_value=1, step=1),
                "end_month": st.column_config.NumberColumn("End Month", min_value=1, step=1),
                "start_members": st.column_config.NumberColumn("Start Members", min_value=0, step=1),
                "end_members": st.column_config.NumberColumn("End Members", min_value=0, step=1),
                "type": st.column_config.SelectboxColumn("Growth Type", options=["linear", "exponential"])
            },
            hide_index=True
        )
        
        params_state["MEMBERSHIP_SEGMENTS"] = edited_segments.to_dict("records")
        
        # Generate and show preview
        months = params_state.get("MONTHS", 60)
        membership_curve = generate_membership_curve_from_segments(edited_segments.to_dict("records"), months)
        
        if membership_curve:
            preview_df = pd.DataFrame({
                "Month": list(range(1, len(membership_curve) + 1)),
                "Members": membership_curve
            })
            st.line_chart(preview_df.set_index("Month")["Members"])
            
            # Store generated curve for later use
            params_state["GENERATED_MEMBERSHIP_CURVE"] = membership_curve
    
    elif params_state["MEMBERSHIP_MODE"] == "calculated":
        st.markdown("**Market Dynamics Calculation**")
        st.caption("Membership calculated from market pools, acquisition rates, and churn - the original model approach.")
        st.info("Configure market dynamics and member behavior in the sections below.")
    
    return params_state

def generate_membership_curve_from_segments(segments: list, total_months: int) -> list:
    """Generate membership curve from piecewise segments"""
    import numpy as np
    
    curve = [0] * total_months
    
    for segment in segments:
        start_month = int(segment["start_month"]) - 1  # Convert to 0-based
        end_month = int(segment["end_month"]) - 1
        start_members = float(segment["start_members"])
        end_members = float(segment["end_members"])
        growth_type = segment["type"]
        
        # Validate bounds
        start_month = max(0, min(start_month, total_months - 1))
        end_month = max(start_month, min(end_month, total_months - 1))
        
        if start_month >= end_month:
            continue
            
        segment_length = end_month - start_month + 1
        
        if growth_type == "linear":
            values = np.linspace(start_members, end_members, segment_length)
        elif growth_type == "exponential":
            if start_members > 0:
                # Exponential growth from start_members to end_members
                growth_rate = (end_members / start_members) ** (1 / (segment_length - 1))
                values = [start_members * (growth_rate ** i) for i in range(segment_length)]
            else:
                # Fall back to linear if starting from 0
                values = np.linspace(start_members, end_members, segment_length)
        else:
            values = np.linspace(start_members, end_members, segment_length)
        
        # Fill in the curve
        for i, val in enumerate(values):
            if start_month + i < total_months:
                curve[start_month + i] = max(0, int(val))
    
    return curve

def render_parameter_group(group_name: str, group_info: dict, params_state: dict) -> dict:
    """Render a logical group of parameters - all parameters shown directly without nested sections"""
    
    # Get parameters for this group
    group_params = {k: v for k, v in COMPLETE_PARAM_SPECS.items() if v.get("group") == group_name}
    
    if not group_params:
        return params_state
    
    # Group header
    color_indicator = {"green": "🟢", "amber": "🟡", "red": "🔴", "blue": "🔵"}.get(group_info["color"], "⚪")
    st.markdown(f"**{group_info['title']}**")
    st.caption(f"{color_indicator} {group_info['desc']}")
    
    # Show all parameters for this group directly (no nested advanced sections)
    for param_name in sorted(group_params.keys()):
        spec = COMPLETE_PARAM_SPECS[param_name]
        params_state[param_name] = render_single_parameter(param_name, spec, params_state.get(param_name), params_state)
    
    return params_state

def render_single_parameter(param_name: str, spec: dict, current_value: Any, params_state: dict) -> Any:
    """Render individual parameter with appropriate Streamlit widget"""
    
    param_type = spec["type"]
    label = spec["label"]
    desc = spec["desc"]
    
    # Set default if no current value
    if current_value is None:
        current_value = spec.get("default", get_param_default(spec))
    
        # Suggestion logic for loan amount overrides
    if param_name in ("LOAN_504_AMOUNT_OVERRIDE", "LOAN_7A_AMOUNT_OVERRIDE"):
        try:
            if param_name == "LOAN_504_AMOUNT_OVERRIDE":
                capex_items = params_state.get("CAPEX_ITEMS", []) or []
                base_504 = sum(
                    float(item.get("unit_cost", 0)) * float(item.get("count", 1))
                    for item in capex_items
                    if (item.get("enabled", True) and item.get("finance_504", True))
                )
                contingency = float(params_state.get("LOAN_CONTINGENCY_PCT", 0.08) or 0.0)
                extra_504 = float(params_state.get("EXTRA_504_BUFFER", 0.0) or 0.0)
                suggested = int(round(base_504 * (1.0 + contingency) + extra_504))
            else:
                monthly_rent = float(params_state.get("RENT", 3500) or 0.0)
                monthly_owner_draw = float(params_state.get("OWNER_DRAW", 2000) or 0.0)
                monthly_insurance = float(params_state.get("INSURANCE_COST", 75) or 0.0)
                monthly_base_opex = monthly_rent + monthly_owner_draw + monthly_insurance
                runway_months = int(params_state.get("RUNWAY_MONTHS", 8) or 8)
                extra_buffer = float(params_state.get("EXTRA_BUFFER", 10000) or 0.0)
                suggested = int(round(monthly_base_opex * runway_months + extra_buffer))
            # Only prefill when value is unset or explicitly zero
            if current_value is None or (isinstance(current_value, (int, float)) and float(current_value) == 0.0):
                current_value = suggested
            # Append suggestion to the tooltip
            desc = f"{desc} Suggested: ${suggested:,.0f} based on current CapEx/OpEx."
        except Exception:
            pass

    # Create help text
    help_text = f"{desc}"
    if "min" in spec and "max" in spec:
        help_text += f" Range: {spec['min']}-{spec['max']}"
    
    # Render appropriate widget
    if param_type == "bool":
        return st.checkbox(label, value=current_value, help=help_text)
    
    elif param_type == "int":
        return st.slider(
            label,
            min_value=int(spec["min"]),
            max_value=int(spec["max"]),
            value=int(current_value),
            step=int(spec.get("step", 1)),
            help=help_text
        )
    
    elif param_type == "float":
        return st.slider(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(current_value),
            step=float(spec.get("step", 0.01)),
            help=help_text,
            format="%.3f" if spec.get("step", 0.01) < 0.01 else "%.2f"
        )
    
    elif param_type == "select":
        options = spec["options"]
        try:
            current_index = options.index(current_value) if current_value in options else 0
        except (ValueError, TypeError):
            current_index = 0
        
        return st.selectbox(
            label,
            options=options,
            index=current_index,
            help=help_text
        )
    
    elif param_type == "text":
        return st.text_input(
            label,
            value=str(current_value),
            help=help_text
        )
    
    return current_value

def get_param_default(spec: dict) -> Any:
    """Get appropriate default value for parameter spec"""
    if "default" in spec:
        return spec["default"]
    elif spec["type"] == "bool":
        return False
    elif spec["type"] in ["int", "float"]:
        return spec.get("min", 0)
    elif spec["type"] == "select":
        return spec["options"][0] if spec.get("options") else None
    elif spec["type"] == "text":
        return ""
    return None

def build_complete_overrides(params_state: dict) -> dict:
    """Convert UI parameter state to simulator overrides with proper mapping"""
    
    overrides = {}
    
    # Handle membership trajectory mode
    membership_mode = params_state.get("MEMBERSHIP_MODE", "calculated")
    overrides["MEMBERSHIP_MODE"] = membership_mode
    
    if membership_mode == "manual_table":
        # Use manual membership table
        manual_table = params_state.get("MANUAL_MEMBERSHIP_TABLE", [])
        overrides["MANUAL_MEMBERSHIP_CURVE"] = manual_table
        # Disable market dynamics when using manual curve
        overrides["USE_MANUAL_MEMBERSHIP_CURVE"] = True
        
    elif membership_mode == "piecewise_trends":
        # Use generated curve from piecewise segments
        generated_curve = params_state.get("GENERATED_MEMBERSHIP_CURVE", [])
        overrides["MANUAL_MEMBERSHIP_CURVE"] = generated_curve
        overrides["USE_MANUAL_MEMBERSHIP_CURVE"] = True
        
    else:  # calculated mode
        overrides["USE_MANUAL_MEMBERSHIP_CURVE"] = False
    
    # Direct parameter mappings (most parameters)
    for param_name, value in params_state.items():
        if param_name in COMPLETE_PARAM_SPECS:
            overrides[param_name] = value
    
    # Special handling for complex parameters
    
    # Member archetype probabilities -> MEMBER_ARCHETYPES structure
    if all(k in params_state for k in ["HOBBYIST_PROB", "COMMITTED_ARTIST_PROB", "PRODUCTION_POTTER_PROB", "SEASONAL_USER_PROB"]):
        overrides["MEMBER_ARCHETYPES"] = {
            "Hobbyist": {
                "prob": params_state["HOBBYIST_PROB"],
                "monthly_fee": params_state.get("PRICE", 175),
                "clay_bags": (
                    params_state.get("HOBBYIST_CLAY_LOW", 0.25),
                    params_state.get("HOBBYIST_CLAY_TYPICAL", 0.5),
                    params_state.get("HOBBYIST_CLAY_HIGH", 1.0)
                )
            },
            "Committed Artist": {
                "prob": params_state["COMMITTED_ARTIST_PROB"],
                "monthly_fee": params_state.get("PRICE", 185),
                "clay_bags": (
                    params_state.get("COMMITTED_ARTIST_CLAY_LOW", 1.0),
                    params_state.get("COMMITTED_ARTIST_CLAY_TYPICAL", 1.5),
                    params_state.get("COMMITTED_ARTIST_CLAY_HIGH", 2.0)
                )
            },
            "Production Potter": {
                "prob": params_state["PRODUCTION_POTTER_PROB"],
                "monthly_fee": params_state.get("PRICE", 200),
                "clay_bags": (
                    params_state.get("PRODUCTION_POTTER_CLAY_LOW", 2.0),
                    params_state.get("PRODUCTION_POTTER_CLAY_TYPICAL", 2.5),
                    params_state.get("PRODUCTION_POTTER_CLAY_HIGH", 3.0)
                )
            },
            "Seasonal User": {
                "prob": params_state["SEASONAL_USER_PROB"],
                "monthly_fee": params_state.get("PRICE", 150),
                "clay_bags": (
                    params_state.get("SEASONAL_USER_CLAY_LOW", 0.25),
                    params_state.get("SEASONAL_USER_CLAY_TYPICAL", 0.5),
                    params_state.get("SEASONAL_USER_CLAY_HIGH", 1.0)
                )
            },
        }
    
    # Churn rates by archetype
    if all(k in params_state for k in ["ARCHETYPE_CHURN_HOBBYIST", "ARCHETYPE_CHURN_COMMITTED_ARTIST", "ARCHETYPE_CHURN_PRODUCTION_POTTER", "ARCHETYPE_CHURN_SEASONAL_USER"]):
        overrides["ARCHETYPE_MONTHLY_CHURN"] = {
            "Hobbyist": params_state["ARCHETYPE_CHURN_HOBBYIST"],
            "Committed Artist": params_state["ARCHETYPE_CHURN_COMMITTED_ARTIST"],
            "Production Potter": params_state["ARCHETYPE_CHURN_PRODUCTION_POTTER"],
            "Seasonal User": params_state["ARCHETYPE_CHURN_SEASONAL_USER"],
        }
    
    # Sessions per week by archetype
    if all(k in params_state for k in ["HOBBYIST_SESSIONS_PER_WEEK", "COMMITTED_ARTIST_SESSIONS_PER_WEEK", "PRODUCTION_POTTER_SESSIONS_PER_WEEK", "SEASONAL_USER_SESSIONS_PER_WEEK"]):
        overrides["SESSIONS_PER_WEEK"] = {
            "Hobbyist": params_state["HOBBYIST_SESSIONS_PER_WEEK"],
            "Committed Artist": params_state["COMMITTED_ARTIST_SESSIONS_PER_WEEK"],
            "Production Potter": params_state["PRODUCTION_POTTER_SESSIONS_PER_WEEK"],
            "Seasonal User": params_state["SEASONAL_USER_SESSIONS_PER_WEEK"],
        }
    
    # Session hours by archetype  
    if all(k in params_state for k in ["HOBBYIST_SESSION_HOURS", "COMMITTED_ARTIST_SESSION_HOURS", "PRODUCTION_POTTER_SESSION_HOURS", "SEASONAL_USER_SESSION_HOURS"]):
        overrides["SESSION_HOURS"] = {
            "Hobbyist": params_state["HOBBYIST_SESSION_HOURS"],
            "Committed Artist": params_state["COMMITTED_ARTIST_SESSION_HOURS"],
            "Production Potter": params_state["PRODUCTION_POTTER_SESSION_HOURS"],
            "Seasonal User": params_state["SEASONAL_USER_SESSION_HOURS"],
        }
    
    # Station capacities and utilization
    if all(k in params_state for k in ["WHEELS_CAPACITY", "HANDBUILDING_CAPACITY", "GLAZE_CAPACITY"]):
        overrides["STATIONS"] = {
            "wheels": {
                "capacity": params_state["WHEELS_CAPACITY"],
                "alpha": params_state.get("WHEELS_ALPHA", 0.80),
                "kappa": 2  # Default from original code
            },
            "handbuilding": {
                "capacity": params_state["HANDBUILDING_CAPACITY"],
                "alpha": params_state.get("HANDBUILDING_ALPHA", 0.50),
                "kappa": 3.0  # Default from original code
            },
            "glaze": {
                "capacity": params_state["GLAZE_CAPACITY"],
                "alpha": params_state.get("GLAZE_ALPHA", 0.55),
                "kappa": 2.6  # Default from original code
            }
        }
    
    # Seasonality array
    seasonality_keys = [f"SEASONALITY_{month}" for month in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
    if all(k in params_state for k in seasonality_keys):
        overrides["SEASONALITY_WEIGHTS"] = np.array([
            params_state["SEASONALITY_JAN"], params_state["SEASONALITY_FEB"], params_state["SEASONALITY_MAR"],
            params_state["SEASONALITY_APR"], params_state["SEASONALITY_MAY"], params_state["SEASONALITY_JUN"],
            params_state["SEASONALITY_JUL"], params_state["SEASONALITY_AUG"], params_state["SEASONALITY_SEP"],
            params_state["SEASONALITY_OCT"], params_state["SEASONALITY_NOV"], params_state["SEASONALITY_DEC"]
        ])
    
    # Market pools and inflows (only needed for calculated mode)
    if membership_mode == "calculated":
        if all(k in params_state for k in ["NO_ACCESS_POOL", "HOME_POOL", "COMMUNITY_POOL"]):
            overrides["MARKET_POOLS"] = {
                "no_access": params_state["NO_ACCESS_POOL"],
                "home_studio": params_state["HOME_POOL"],
                "community_studio": params_state["COMMUNITY_POOL"]
            }
        
        if all(k in params_state for k in ["NO_ACCESS_INFLOW", "HOME_INFLOW", "COMMUNITY_INFLOW"]):
            overrides["MARKET_POOLS_INFLOW"] = {
                "no_access": params_state["NO_ACCESS_INFLOW"],
                "home_studio": params_state["HOME_INFLOW"],
                "community_studio": params_state["COMMUNITY_INFLOW"]
            }
        
        if all(k in params_state for k in ["BASELINE_RATE_NO_ACCESS", "BASELINE_RATE_HOME", "BASELINE_RATE_COMMUNITY"]):
            overrides["POOL_BASE_INTENT"] = {
                "no_access": params_state["BASELINE_RATE_NO_ACCESS"],
                "home_studio": params_state["BASELINE_RATE_HOME"],  
                "community_studio": params_state["BASELINE_RATE_COMMUNITY"]
            }
    
    # Handle JSON text parameters (events)
    if "ATTENDEES_PER_EVENT_RANGE" in params_state:
        try:
            overrides["ATTENDEES_PER_EVENT_RANGE"] = json.loads(params_state["ATTENDEES_PER_EVENT_RANGE"])
        except (json.JSONDecodeError, TypeError):
            overrides["ATTENDEES_PER_EVENT_RANGE"] = [8, 10, 12]  # Default
    
    if "EVENT_MUG_COST_RANGE" in params_state:
        try:
            overrides["EVENT_MUG_COST_RANGE"] = tuple(json.loads(params_state["EVENT_MUG_COST_RANGE"]))
        except (json.JSONDecodeError, TypeError):
            overrides["EVENT_MUG_COST_RANGE"] = (4.5, 7.5)  # Default
    
    # Economic environment parameters that need to be at top level
    # (these are used directly by the simulator, not within SCENARIO_CONFIGS)
    economic_params = [
        "DOWNTURN_PROB_PER_MONTH", "DOWNTURN_JOIN_MULT", "DOWNTURN_CHURN_MULT",
        "WOM_Q", "WOM_SATURATION", "REFERRAL_RATE_PER_MEMBER", "REFERRAL_CONV",
        "AWARENESS_RAMP_MONTHS", "AWARENESS_RAMP_START_MULT", "AWARENESS_RAMP_END_MULT",
        "ADOPTION_SIGMA", "CLASS_TERM_MONTHS", "CS_UNLOCK_FRACTION_PER_TERM",
        "MAX_ONBOARDINGS_PER_MONTH", "CAPACITY_DAMPING_BETA", "UTILIZATION_CHURN_UPLIFT"
    ]
    
    for param in economic_params:
        if param in params_state:
            overrides[param] = params_state[param]

    # ---------- Loan wiring: UI -> simulator globals ----------
    # Forward core 7(a)/504 knobs so simulator doesn't use defaults
    _loan_keys = [
        "RUNWAY_MONTHS",
        "LOAN_504_ANNUAL_RATE", "LOAN_504_TERM_YEARS", "IO_MONTHS_504",
        "LOAN_7A_ANNUAL_RATE",  "LOAN_7A_TERM_YEARS",  "IO_MONTHS_7A",
        "LOAN_CONTINGENCY_PCT", "EXTRA_BUFFER",
        "FEES_UPFRONT_PCT_7A", "FEES_UPFRONT_PCT_504",
        "FEES_PACKAGING", "FEES_CLOSING",
        "FINANCE_FEES_7A", "FINANCE_FEES_504",
        "RESERVE_FLOOR",
    ]
    for k in _loan_keys:
        if k in params_state:
            overrides[k] = params_state[k]

    # Map UI principal overrides to simulator names
    # UI: LOAN_504_AMOUNT_OVERRIDE / LOAN_7A_AMOUNT_OVERRIDE
    # SIM: LOAN_OVERRIDE_504 / LOAN_OVERRIDE_7A
    try:
        ov504 = float(params_state.get("LOAN_504_AMOUNT_OVERRIDE", 0) or 0)
        if ov504 > 0:
            overrides["LOAN_OVERRIDE_504"] = ov504
    except Exception:
        pass
    try:
        ov7a = float(params_state.get("LOAN_7A_AMOUNT_OVERRIDE", 0) or 0)
        if ov7a > 0:
            overrides["LOAN_OVERRIDE_7A"] = ov7a
    except Exception:
        pass

    
    # Scenario config wrapper (required by simulator)
    grant_amount = params_state.get("grant_amount", 0.0)
    grant_month = params_state.get("grant_month", -1)
    grant_month = None if grant_month == -1 else grant_month
    
    overrides["SCENARIO_CONFIGS"] = [{
        "name": "User_Defined",
        "grant_amount": grant_amount,
        "grant_month": grant_month
    }]
    
    return overrides

# UI MAIN INTERFACE
def render_complete_ui():
    """Render the complete parameter interface"""
    
    st.set_page_config(page_title="GCWS Complete Scenario Planner", layout="wide")
    st.title("🏺 Ginkgo Clayworks Studio - Complete Scenario Planner")
    
    # Warning banner
    st.error("""
    ⚠️ **SCENARIO PLANNING TOOL - NOT PREDICTIVE**
    
    This tool explores business scenarios based on YOUR assumptions. Results depend entirely on the parameters you set below.
    This is NOT a prediction of actual business performance. Use for exploring possibilities and understanding sensitivities.
    """)
    
    # Initialize session state
    if "params_state" not in st.session_state:
        # Initialize with defaults
        st.session_state.params_state = {}
        for param_name, spec in COMPLETE_PARAM_SPECS.items():
            st.session_state.params_state[param_name] = spec.get("default", get_param_default(spec))
    
    # Group visibility and reset (moved from sidebar)
    st.subheader("View")
    show_all_groups = st.checkbox("Show all parameter groups", value=False)
    if not show_all_groups:
        selected_groups = st.multiselect(
            "Select parameter groups to show:",
            options=list(PARAMETER_GROUPS.keys()),
            default=["business_fundamentals", "pricing", "capacity"],
            format_func=lambda x: PARAMETER_GROUPS[x]["title"]
        )
    else:
        selected_groups = list(PARAMETER_GROUPS.keys())

    # Reset to defaults button
    if st.button("Reset All to Defaults"):
        st.session_state.params_state = {}
        for param_name, spec in COMPLETE_PARAM_SPECS.items():
            st.session_state.params_state[param_name] = spec.get("default", get_param_default(spec))
        st.experimental_rerun()

    # Main parameter interface
    st.header("Parameter Configuration")
    st.markdown("Each parameter includes a tooltip explaining its impact on the model. Adjust values based on your specific situation and market research.")
    
    # Render selected parameter groups
    groups_by_priority = sorted(
        [(k, v) for k, v in PARAMETER_GROUPS.items() if k in selected_groups],
        key=lambda x: x[1]["priority"]
    )
    
    for group_name, group_info in groups_by_priority:
        with st.expander(group_info["title"], expanded=(group_info["priority"] <= 4)):
            if group_name == "membership_trajectory":
                # Special handling for membership trajectory
                st.session_state.params_state = render_membership_trajectory(st.session_state.params_state)
            else:
                st.session_state.params_state = render_parameter_group(
                    group_name, group_info, st.session_state.params_state
                )
                # Financing group convenience actions
                if group_name == "financing":
                    st.caption("Tip: Click below to refresh the 504 and 7(a) fields with current suggestions.")
                    if st.button("Reset loan asks to suggestions", key="btn_reset_loan_suggestions"):
                        st.session_state.params_state["LOAN_504_AMOUNT_OVERRIDE"] = 0.0
                        st.session_state.params_state["LOAN_7A_AMOUNT_OVERRIDE"] = 0.0
                        st.experimental_rerun()
    
    # Equipment configuration (special handling)
    with st.expander("🔧 Staff payroll Expenditures", expanded=False):
        st.markdown("**Staff hiring schedule**")
        st.caption("Define when staff are hired, their compensation, and duration of employment.")
        
        # Default staff configuration
        default_staff = [
            {"enabled": False, "role": "Part-time Assistant", "start_month": 6, "end_month": None, "hourly_rate": 18.0, "hours_per_week": 20, "trigger_members": None},
            {"enabled": False, "role": "Studio Manager", "start_month": 12, "end_month": None, "hourly_rate": 25.0, "hours_per_week": 30, "trigger_members": None},
            {"enabled": False, "role": "Evening Instructor", "start_month": None, "end_month": None, "hourly_rate": 30.0, "hours_per_week": 15, "trigger_members": 50},
        ]
        
        if "STAFF_SCHEDULE" not in st.session_state.params_state:
            st.session_state.params_state["STAFF_SCHEDULE"] = default_staff
        
        staff_df = pd.DataFrame(st.session_state.params_state["STAFF_SCHEDULE"])
        
        edited_staff = st.data_editor(
            staff_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Include", help="Whether this staff position is filled"),
                "role": st.column_config.TextColumn("Role/Title", help="Staff position description"),
                "start_month": st.column_config.NumberColumn("Start Month", min_value=0, step=1, help="Month to hire (0=immediate, leave blank for member-triggered)"),
                "end_month": st.column_config.NumberColumn("End Month", min_value=1, step=1, help="Last month of employment (blank=permanent through forecast)"),
                "hourly_rate": st.column_config.NumberColumn("Hourly Rate ($)", min_value=10.0, step=0.50, help="Hourly compensation including taxes/benefits"),
                "hours_per_week": st.column_config.NumberColumn("Hours/Week", min_value=1.0, step=1.0, help="Average hours worked per week"),
                "trigger_members": st.column_config.NumberColumn("Member Trigger", min_value=0, step=1, help="Member count to trigger hiring (blank for month-based)")
            }
        )
        
        st.session_state.params_state["STAFF_SCHEDULE"] = edited_staff.to_dict("records")
        
        # Show calculated monthly costs
        if len(edited_staff) > 0:
            enabled_staff = edited_staff[edited_staff.get("enabled", False) == True]
            if len(enabled_staff) > 0:
                total_monthly_cost = sum(
                    (row.get("hourly_rate", 0) * row.get("hours_per_week", 0) * 52 / 12)
                    for _, row in enabled_staff.iterrows()
                )
                st.info(f"Total monthly staff cost when all enabled positions are active: ${total_monthly_cost:,.0f}")
    
    with st.expander("🔧 Equipment & Capital Expenditures", expanded=False):
        st.markdown("**Equipment purchase schedule**")
        st.caption("Define when equipment is purchased (by month or member count) and whether it's financed through SBA 504 loans.")
        
        # Default equipment configuration
        default_capex = [
            {"enabled": True,  "label": "Kiln #1 Skutt 1227", "count": 1,  "unit_cost": 7000, "month": 0,    "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Pottery Wheels",     "count": 4,  "unit_cost": 800,  "month": 0,    "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Wire Racks",         "count": 10, "unit_cost": 100,  "month": 0,    "member_threshold": None, "finance_504": True},
            {"enabled": True,  "label": "Clay Traps",         "count": 1,  "unit_cost": 400,  "month": 0,    "member_threshold": None, "finance_504": True},
            {"enabled": False, "label": "Kiln #2 Skutt 1427", "count": 1,  "unit_cost": 9000, "month": 6,    "member_threshold": None, "finance_504": True},
            {"enabled": False, "label": "Slab Roller",        "count": 1,  "unit_cost": 3000, "month": None, "member_threshold": 50,   "finance_504": True},
            {"enabled": False, "label": "Pug Mill",           "count": 1,  "unit_cost": 4500, "month": None, "member_threshold": 75,   "finance_504": True},
         ]
        
        if "CAPEX_ITEMS" not in st.session_state.params_state:
            st.session_state.params_state["CAPEX_ITEMS"] = default_capex
        
        # Equipment data editor
        capex_df = pd.DataFrame(st.session_state.params_state["CAPEX_ITEMS"])
        
        edited_df = st.data_editor(
            capex_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "enabled": st.column_config.CheckboxColumn("Include", help="Whether this equipment is purchased"),
                "label": st.column_config.TextColumn("Equipment", help="Equipment description"),
                "count": st.column_config.NumberColumn("Quantity", min_value=1, step=1, help="Number of units"),
                "unit_cost": st.column_config.NumberColumn("Unit Cost ($)", min_value=0, step=100, help="Cost per unit"),
                "month": st.column_config.NumberColumn("Trigger Month", min_value=0, step=1, help="Month to purchase (0=immediate, leave blank for member-based)"),
                "member_threshold": st.column_config.NumberColumn("Member Threshold", min_value=0, step=1, help="Member count to trigger purchase (leave blank for month-based)"),
                "finance_504": st.column_config.CheckboxColumn("SBA 504", help="Finance through SBA 504 loan")
            }
        )
        
        # Update session state
        st.session_state.params_state["CAPEX_ITEMS"] = edited_df.to_dict("records")

    # Firing fee schedule (special handling)
    with st.expander("🔥 Firing Fee Schedule (per-lb tiers)", expanded=False):
        st.markdown("**Define a tiered per-lb firing fee schedule**")
        st.caption("Each row is a tier. 'Up to lbs' is the upper bound for that tier (leave blank for the last, open-ended tier). 'Rate' is $/lb.")
        # Default schedule
        default_sched = [
            {"up_to_lbs": 20, "rate": 3.0},
            {"up_to_lbs": 40, "rate": 4.0},
            {"up_to_lbs": None, "rate": 5.0},
        ]
        if "FIRING_FEE_SCHEDULE" not in st.session_state.params_state:
            st.session_state.params_state["FIRING_FEE_SCHEDULE"] = default_sched
        sched_df = pd.DataFrame(st.session_state.params_state["FIRING_FEE_SCHEDULE"])
        edited_sched = st.data_editor(
            sched_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "up_to_lbs": st.column_config.NumberColumn("Up to lbs", min_value=0, step=1, help="Upper bound for this tier (blank = no upper bound)"),
                "rate": st.column_config.NumberColumn("Rate ($/lb)", min_value=0.0, step=0.5, help="Charge per lb within this tier"),
            }
        )
        # Clean and validate
        recs = edited_sched.to_dict("records")
        cleaned = []
        last = -1
        for r in recs:
            up = r.get("up_to_lbs", None)
            if up == "" or up is None:
                up = None
            else:
                try:
                    up = int(up)
                except Exception:
                    up = None
            rate = r.get("rate", None)
            try:
                rate = float(rate) if rate is not None else None
            except Exception:
                rate = None
            if rate is None:
                continue
            # enforce strictly increasing bounds
            if up is not None and up <= last:
                up = last + 1
            cleaned.append({"up_to_lbs": up, "rate": rate})
            if up is not None:
                last = up
        # Ensure final open tier
        if cleaned and cleaned[-1]["up_to_lbs"] is not None:
            cleaned.append({"up_to_lbs": None, "rate": cleaned[-1]["rate"]})
        st.session_state.params_state["FIRING_FEE_SCHEDULE"] = cleaned

    # Run form at bottom
    with st.form("run_form"):
        st.subheader("Run simulation")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.params_state["MONTHS"] = st.slider(
                "Simulation horizon (months)", 12, 120,
                int(st.session_state.params_state.get("MONTHS", 60)), 6
            )
        with c2:
            st.session_state.params_state["N_SIMULATIONS"] = st.slider(
                "Monte Carlo draws", 1, 2000,
                int(st.session_state.params_state.get("N_SIMULATIONS", 100)), 1
            )
        with c3:
            st.session_state.params_state["RANDOM_SEED"] = st.number_input(
                "Random seed", 1, 999999,
                int(st.session_state.params_state.get("RANDOM_SEED", 42))
            )

        run_simulation = st.form_submit_button("🚀 Run Simulation", type="primary")
    
    # Run simulation
    if run_simulation:
        # Clear any existing cache to ensure fresh run
        try:
            st.cache_data.clear()
        except:
            pass
            
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Build overrides from UI state
                overrides = build_complete_overrides(st.session_state.params_state)
                
                # Add equipment items
                if "CAPEX_ITEMS" in st.session_state.params_state:
                    overrides["CAPEX_ITEMS"] = _normalize_capex_items(pd.DataFrame(st.session_state.params_state["CAPEX_ITEMS"]))
                
                # Add firing fee schedule
                if "FIRING_FEE_SCHEDULE" in st.session_state.params_state:
                    # Pass as a native list of dicts; simulator also supports JSON string
                    import json as _json  # available for any future JSON encoding
                    overrides["FIRING_FEE_SCHEDULE"] = st.session_state.params_state["FIRING_FEE_SCHEDULE"]

                # Run simulation with figure capture
                with FigureCapture("User Defined Scenario") as cap:
                    results = run_original_once("modular_simulator.py", overrides)
                
                if isinstance(results, tuple):
                    df, eff = results
                else:
                    df = results
                    eff = None
                
                if df is None or df.empty:
                    st.error("Simulation returned no results. Check parameter values and try again.")
                    return
                
                # Store results in session state
                st.session_state["simulation_results"] = df
                st.session_state["simulation_images"] = cap.images
                st.session_state["simulation_manifest"] = cap.manifest
                
                # Display results
                st.success(f"Simulation completed: {len(df)} result rows generated")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                final_month = df["month"].max()
                final_data = df[df["month"] == final_month]
                
                # Survival rate - check if any simulation went negative
                survival_rate = (df.groupby("simulation_id")["cash_balance"].min() >= 0).mean()
                
                col1.metric("Survival Rate", f"{survival_rate:.1%}")
                col2.metric("Median Final Cash", f"${final_data['cash_balance'].median():,.0f}")
                
                if "active_members" in df.columns:
                    col3.metric("Median Final Members", f"{final_data['active_members'].median():.0f}")
                
                if "dscr" in df.columns:
                    final_dscr = final_data["dscr"].replace([np.inf, -np.inf], np.nan)
                    col4.metric("Median Final DSCR", f"{final_dscr.median():.2f}")
                
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)
                return
    
    # Display results if available
    if "simulation_results" in st.session_state:
        df = st.session_state["simulation_results"]
        images = st.session_state.get("simulation_images", [])
        manifest = st.session_state.get("simulation_manifest", [])
        
        # Display captured visualizations
        if images:
            st.header("📊 Simulation Analysis Charts")
            st.markdown("The following charts were generated during the simulation analysis:")
            
            for fname, img_data in images:
                # Find corresponding manifest entry for title
                img_title = fname
                for entry in manifest:
                    if entry["file"] == fname:
                        img_title = entry.get("title", fname)
                        break
                
                st.image(img_data, caption=img_title, use_container_width=True)
            
            # Download bundle
            if images:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
                    for fname, data in images:
                        zf.writestr(fname, data)
                
                st.download_button(
                    "📦 Download All Charts (ZIP)",
                    data=buf.getvalue(),
                    file_name=f"gcws_charts_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
        
        # Enhanced Loan Amount Displays
        with st.expander("💰 Loan Amount Summary", expanded=True):
            # --- NEW: ensure loan metrics exist in this scope
            params_state = st.session_state.params_state
            loan_metrics = calculate_loan_metrics(df, params_state)

            # --- NEW: prefill session with calculated values, but allow overrides
            calc_504 = int(loan_metrics.get("total_504_amount", 0) or 0)
            calc_7a  = int(loan_metrics.get("total_7a_amount", 0) or 0)
            # Effective principals used in results: override if >0 else suggestion
            override_504 = int(params_state.get("LOAN_504_AMOUNT_OVERRIDE", 0) or 0)
            override_7a  = int(params_state.get("LOAN_7A_AMOUNT_OVERRIDE", 0) or 0)
            used_504 = override_504 if override_504 > 0 else calc_504
            used_7a  = override_7a  if override_7a  > 0 else calc_7a

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SBA 504 Loan (CapEx)")
                # READ-ONLY display of the amount used (no +/- spinner)
                st.caption("SBA 504 Total Loan Ask ($)")
                st.markdown(f"### ${used_504:,.0f}")
                st.metric(
                    "Used Loan Amount (Override)" if override_504 > 0 else "Used Loan Amount (Auto-calc)",
                    f"${used_504:,.0f}"
                )
                st.metric("Interest-Only Payment", f"${loan_metrics['io_payment_504']:,.0f}/month")
                st.metric("Amortizing Payment", f"${loan_metrics['amort_payment_504']:,.0f}/month")
            
            with col2:
                st.subheader("SBA 7(a) Loan (OpEx)")
                # READ-ONLY display of the amount used (no +/- spinner)
                st.caption("SBA 7(a) Total Loan Ask ($)")
                st.markdown(f"### ${used_7a:,.0f}")
                st.metric(
                    "Used Loan Amount (Override)" if override_7a > 0 else "Used Loan Amount (Auto-calc)",
                    f"${used_7a:,.0f}"
                )
                st.metric("Interest-Only Payment", f"${loan_metrics['io_payment_7a']:,.0f}/month")
                st.metric("Amortizing Payment", f"${loan_metrics['amort_payment_7a']:,.0f}/month")
            
            
            # Keep legacy variable names for code below
            total_504 = used_504
            total_7a  = used_7a

            # Total debt service
            st.subheader("Combined Debt Service")
            total_io = loan_metrics['io_payment_504'] + loan_metrics['io_payment_7a']
            total_amort = loan_metrics['amort_payment_504'] + loan_metrics['amort_payment_7a']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Debt", f"${total_504 + total_7a:,.0f}")
            col2.metric("IO Period Payment", f"${total_io:,.0f}/month")
            col3.metric("Amortizing Payment", f"${total_amort:,.0f}/month")
        st.header("📈 Detailed Results")
        
        # Compute key statistics by simulation
        summary_stats = []
        for sim_id in df["simulation_id"].unique():
            sim_data = df[df["simulation_id"] == sim_id]
            
            # Key metrics per simulation
            min_cash = sim_data["cash_balance"].min()
            final_cash = sim_data[sim_data["month"] == sim_data["month"].max()]["cash_balance"].iloc[0]
            final_members = sim_data[sim_data["month"] == sim_data["month"].max()]["active_members"].iloc[0] if "active_members" in sim_data.columns else 0
            
            # Break-even analysis
            cumulative_profit = sim_data.get("cumulative_op_profit", pd.Series([0]))
            breakeven_month = cumulative_profit[cumulative_profit >= 0].index
            breakeven_month = sim_data.loc[breakeven_month].iloc[0]["month"] if len(breakeven_month) > 0 else None
            
            summary_stats.append({
                "simulation_id": sim_id,
                "min_cash": min_cash,
                "final_cash": final_cash,
                "final_members": final_members,
                "breakeven_month": breakeven_month,
                "survival": min_cash >= 0
            })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Display key percentiles
        st.subheader("Key Risk Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("10th Percentile Min Cash", f"${summary_df['min_cash'].quantile(0.1):,.0f}")
            st.metric("50th Percentile Min Cash", f"${summary_df['min_cash'].quantile(0.5):,.0f}")
            st.metric("90th Percentile Min Cash", f"${summary_df['min_cash'].quantile(0.9):,.0f}")
        
        with col2:
            st.metric("10th Percentile Final Cash", f"${summary_df['final_cash'].quantile(0.1):,.0f}")
            st.metric("50th Percentile Final Cash", f"${summary_df['final_cash'].quantile(0.5):,.0f}")
            st.metric("90th Percentile Final Cash", f"${summary_df['final_cash'].quantile(0.9):,.0f}")
            
        with col3:
            if summary_df['breakeven_month'].notna().any():
                be_months = summary_df['breakeven_month'].dropna()
                st.metric("10th Percentile Breakeven", f"{be_months.quantile(0.1):.0f} months" if len(be_months) > 0 else "Never")
                st.metric("50th Percentile Breakeven", f"{be_months.quantile(0.5):.0f} months" if len(be_months) > 0 else "Never")
                st.metric("90th Percentile Breakeven", f"{be_months.quantile(0.9):.0f} months" if len(be_months) > 0 else "Never")
        
        # Enhanced Loan Analysis Section (render_loan_analysis handles its own errors)
        render_loan_analysis(df, st.session_state.params_state)# Enhanced Loan Analysis Section
         
        # Raw data download and display
        st.subheader("Raw Simulation Data")
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "📄 Download Full Results (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"gcws_simulation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_buffer = io.StringIO()
            summary_df.to_csv(summary_buffer, index=False)
            st.download_button(
                "📊 Download Summary Stats (CSV)",
                data=summary_buffer.getvalue(),
                file_name=f"gcws_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Display sample of raw data
        st.dataframe(df.head(200), use_container_width=True)
        
        if len(df) > 200:
            st.caption(f"Showing first 200 of {len(df)} total rows. Download CSV for complete data.")

# HELPER FUNCTIONS FROM ORIGINAL CODE

def _normalize_capex_items(df):
    """Convert equipment dataframe to list of dicts for simulator"""
    items = []
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return items
    
    for _, r in df.iterrows():
        if not r.get("enabled", True):
            continue
        
        try:
            label = str(r.get("label", "")).strip()
            unit = float(r.get("unit_cost", 0) or 0)
            cnt = int(r.get("count", 1) or 1)
            mth = r.get("month", None)
            thr = r.get("member_threshold", None)
            
            # Handle None/NaN values
            mth = None if pd.isna(mth) else int(mth)
            thr = None if pd.isna(thr) else int(thr)
            
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

# Keep existing simulation execution and plotting functions
class FigureCapture:
    """Context manager for capturing matplotlib figures with error handling"""
    def __init__(self, title_suffix: str = ""):
        self.title_suffix = title_suffix
        self._orig_show = None
        self.images: List[Tuple[str, bytes]] = []
        self.manifest = []

    def __enter__(self):
        try:
            matplotlib.use("Agg", force=True)
        except:
            pass  # Backend might already be set
            
        self._orig_show = plt.show
        counter = {"i": 0}

        def _show(*args, **kwargs):
            try:
                counter["i"] += 1
                fig = plt.gcf()
                
                buf = io.BytesIO()
                fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
                buf.seek(0)
                fname = f"fig_{counter['i']:02d}.png"
                self.images.append((fname, buf.read()))
                self.manifest.append({"file": fname, "title": f"Figure {counter['i']}"})
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not capture figure: {e}")

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_show:
            plt.show = self._orig_show

# Main execution
if __name__ == "__main__":
    render_complete_ui()