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
import matplotlib.pyplot as plt
import seaborn as sns
from modular_simulator import get_default_cfg
from final_batch_adapter import run_original_once
from sba_export import export_to_sba_workbook
import os

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
    # SIMULATION SETTINGS
    # =============================================================================
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

# GROUP DEFINITIONS WITH LOGICAL ORGANIZATION
PARAMETER_GROUPS = {
    "business_fundamentals": {
        "title": "🏢 Business Fundamentals", 
        "color": "green",
        "desc": "Core business parameters most likely to vary between studios and locations.",
        "priority": 1
    },
    "pricing": {
        "title": "💰 Pricing & Market Response", 
        "color": "amber", 
        "desc": "Pricing strategy and how customers respond to price changes.",
        "priority": 2
    },
    "member_behavior": {
        "title": "👥 Member Behavior & Archetypes", 
        "color": "amber",
        "desc": "Member mix, usage patterns, and retention characteristics by member type.",
        "priority": 3
    },
    "capacity": {
        "title": "🏭 Studio Capacity & Equipment", 
        "color": "green",
        "desc": "Physical capacity constraints, equipment counts, and utilization factors.",
        "priority": 4
    },
    "market_dynamics": {
        "title": "📈 Market Dynamics & Acquisition", 
        "color": "amber",
        "desc": "Market size, acquisition channels, word-of-mouth, and member acquisition rates.",
        "priority": 5
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

# PARAMETER RENDERING FUNCTIONS
def render_parameter_group(group_name: str, group_info: dict, params_state: dict, show_advanced: bool = False) -> dict:
    """Render a logical group of parameters with progressive disclosure"""
    
    # Get parameters for this group
    group_params = {k: v for k, v in COMPLETE_PARAM_SPECS.items() if v.get("group") == group_name}
    
    if not group_params:
        return params_state
    
    # Sort by parameter importance (basic vs advanced)
    basic_params = []
    advanced_params = []
    
    for param_name, spec in group_params.items():
        # Heuristic: core business params are basic, detailed tuning params are advanced
        if group_name in ["business_fundamentals", "pricing", "capacity", "fixed_costs"] or "ENABLED" in param_name:
            basic_params.append(param_name)
        else:
            advanced_params.append(param_name)
    
    # Group header
    color_indicator = {"green": "🟢", "amber": "🟡", "red": "🔴", "blue": "🔵"}.get(group_info["color"], "⚪")
    st.markdown(f"**{group_info['title']}**")
    st.caption(f"{color_indicator} {group_info['desc']}")
    
    # Always show basic parameters
    for param_name in sorted(basic_params):
        spec = COMPLETE_PARAM_SPECS[param_name]
        params_state[param_name] = render_single_parameter(param_name, spec, params_state.get(param_name))
    
    # Advanced parameters in expander
    if advanced_params:
        with st.expander("🔧 Advanced Settings", expanded=False):
            for param_name in sorted(advanced_params):
                spec = COMPLETE_PARAM_SPECS[param_name]
                params_state[param_name] = render_single_parameter(param_name, spec, params_state.get(param_name))
    
    return params_state

def render_single_parameter(param_name: str, spec: dict, current_value: Any) -> Any:
    """Render individual parameter with appropriate Streamlit widget"""
    
    param_type = spec["type"]
    label = spec["label"]
    desc = spec["desc"]
    
    # Set default if no current value
    if current_value is None:
        current_value = spec.get("default", get_param_default(spec))
    
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
    
    # Market pools and inflows
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
    
    # Sidebar with simulation controls
    with st.sidebar:
        st.header("⚙️ Simulation Controls")
        
        # Core simulation settings (always visible)
        st.session_state.params_state["MONTHS"] = st.slider(
            "Simulation horizon (months)", 12, 120, 
            st.session_state.params_state.get("MONTHS", 60), 6
        )
        st.session_state.params_state["N_SIMULATIONS"] = st.slider(
            "Number of simulations", 10, 500,
            st.session_state.params_state.get("N_SIMULATIONS", 100), 10
        )
        st.session_state.params_state["RANDOM_SEED"] = st.number_input(
            "Random seed", 1, 999999,
            st.session_state.params_state.get("RANDOM_SEED", 42)
        )
        
        # Run button
        run_simulation = st.button("🚀 Run Simulation", type="primary")
        
        st.markdown("---")
        
        # Parameter visibility controls
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
            st.session_state.params_state = render_parameter_group(
                group_name, group_info, st.session_state.params_state
            )
    
    # Equipment configuration (special handling)
    with st.expander("🔧 Equipment & Capital Expenditures", expanded=False):
        st.markdown("**Equipment purchase schedule**")
        st.caption("Define when equipment is purchased (by month or member count) and whether it's financed through SBA 504 loans.")
        
        # Default equipment configuration
        default_capex = [
            {"enabled": True, "label": "Kiln #1 Skutt 1227", "count": 1, "unit_cost": 7000, "month": 0, "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Pottery Wheels", "count": 12, "unit_cost": 800, "month": 0, "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Wire Racks", "count": 15, "unit_cost": 150, "month": 0, "member_threshold": None, "finance_504": True},
            {"enabled": True, "label": "Clay Traps", "count": 4, "unit_cost": 400, "month": 0, "member_threshold": None, "finance_504": True},
            {"enabled": False, "label": "Kiln #2 Skutt 1427", "count": 1, "unit_cost": 10000, "month": 6, "member_threshold": None, "finance_504": True},
            {"enabled": False, "label": "Slab Roller", "count": 1, "unit_cost": 1800, "month": None, "member_threshold": 50, "finance_504": True},
            {"enabled": False, "label": "Pug Mill", "count": 1, "unit_cost": 3500, "month": None, "member_threshold": 75, "finance_504": True},
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
                "member_threshold": st.column_config.NumberColumn("Member Trigger", min_value=0, step=1, help="Member count to trigger purchase (leave blank for month-based)"),
                "finance_504": st.column_config.CheckboxColumn("SBA 504", help="Finance through SBA 504 loan")
            }
        )
        
        # Update session state
        st.session_state.params_state["CAPEX_ITEMS"] = edited_df.to_dict("records")
    
    # Run simulation
    if run_simulation:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Build overrides from UI state
                overrides = build_complete_overrides(st.session_state.params_state)
                
                # Run simulation
                results = run_original_once("modular_simulator.py", overrides)
                
                if isinstance(results, tuple):
                    df, eff = results
                else:
                    df = results
                    eff = None
                
                if df is None or df.empty:
                    st.error("Simulation returned no results. Check parameter values and try again.")
                    return
                
                # Display results
                st.success(f"Simulation completed: {len(df)} result rows generated")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                final_month = df["month"].max()
                final_cash = df[df["month"] == final_month]["cash_balance"]
                survival_rate = (df.groupby("simulation_id")["cash_balance"].min() >= 0).mean()
                
                col1.metric("Survival Rate", f"{survival_rate:.1%}")
                col2.metric("Median Final Cash", f"${final_cash.median():,.0f}")
                
                if "active_members" in df.columns:
                    final_members = df[df["month"] == final_month]["active_members"]
                    col3.metric("Median Final Members", f"{final_members.median():.0f}")
                
                if "dscr" in df.columns:
                    final_dscr = df[df["month"] == final_month]["dscr"]
                    col4.metric("Median Final DSCR", f"{final_dscr.median():.2f}")
                
                # Charts will be captured by existing plotting code in simulator
                st.subheader("Simulation Results")
                st.dataframe(df.head(100))
                
                # Download results
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "Download Results CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"gcws_simulation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

# HELPER FUNCTIONS FROM ORIGINAL CODE
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first available column from candidates list"""
    for c in candidates:
        if c in df.columns:
            return c
    return None

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
    """Context manager for capturing matplotlib figures (from original code)"""
    def __init__(self, title_suffix: str = ""):
        self.title_suffix = title_suffix
        self._orig_show = None
        self.images: List[Tuple[str, bytes]] = []
        self.manifest = []

    def __enter__(self):
        matplotlib.use("Agg", force=True)
        self._orig_show = plt.show
        counter = {"i": 0}

        def _show(*args, **kwargs):
            counter["i"] += 1
            fig = plt.gcf()
            
            buf = io.BytesIO()
            fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
            buf.seek(0)
            fname = f"fig_{counter['i']:02d}.png"
            self.images.append((fname, buf.read()))
            self.manifest.append({"file": fname, "title": f"Figure {counter['i']}"})
            plt.close(fig)

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_show:
            plt.show = self._orig_show

# Main execution
if __name__ == "__main__":
    render_complete_ui()