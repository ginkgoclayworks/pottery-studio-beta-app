#!/usr/bin/env python3
"""
Parameter specifications and mappings for the pottery studio simulator.
Consolidates redundant parameters and provides clean mappings to simulator expectations.
"""

import numpy as np

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

def consolidate_build_overrides(env, strat):
    """
    Maps consolidated UI parameters to simulator expected parameters
    """
    # Combine env and strat into scenario_params
    scenario_params = {}
    if env:
        scenario_params.update(env)
    if strat:
        scenario_params.update(strat)
    
    # Map consolidated parameters to simulator expected parameters
    param_mapping = {
        # Core business parameters that were consolidated
        'MONTHLY_RENT': 'RENT',
        'OWNER_COMPENSATION': 'OWNER_DRAW',
        
        # All other parameters pass through unchanged
        # (Most weren't actually renamed during consolidation)
        'JOIN_PRICE_ELASTICITY': 'JOIN_PRICE_ELASTICITY',
        'CHURN_PRICE_ELASTICITY': 'CHURN_PRICE_ELASTICITY',
        'WOM_RATE': 'WOM_RATE',
        'MARKETING_SPEND': 'MARKETING_SPEND',
        'CAC': 'CAC',
        'LEAD_TO_JOIN_RATE': 'LEAD_TO_JOIN_RATE',
        'STUDIO_CAPACITY': 'STUDIO_CAPACITY',
        'EXPANSION_THRESHOLD': 'EXPANSION_THRESHOLD',
        'MAX_ONBOARD_PER_MONTH': 'MAX_ONBOARD_PER_MONTH',
        'MEMBERSHIP_PRICE': 'MEMBERSHIP_PRICE',
        'REFERENCE_PRICE': 'REFERENCE_PRICE',
        'RENT_GROWTH_PCT': 'RENT_GROWTH_PCT',
        'ECONOMIC_STRESS_LEVEL': 'ECONOMIC_STRESS_LEVEL',
        'DOWNTURN_JOIN_MULT': 'DOWNTURN_JOIN_MULT',
        'DOWNTURN_CHURN_MULT': 'DOWNTURN_CHURN_MULT',
        'MARKET_POOLS_INFLOW': 'MARKET_POOLS_INFLOW',
        'WORKSHOPS_ENABLED': 'WORKSHOPS_ENABLED',
        'WORKSHOPS_PER_MONTH': 'WORKSHOPS_PER_MONTH',
        'WORKSHOP_PRICE': 'WORKSHOP_PRICE',
        'WORKSHOP_CAPACITY': 'WORKSHOP_CAPACITY',
        'WORKSHOP_CONV_RATE': 'WORKSHOP_CONV_RATE',
        'WORKSHOP_CONV_LAG_MO': 'WORKSHOP_CONV_LAG_MO',
        'WORKSHOP_COST_PER_EVENT': 'WORKSHOP_COST_PER_EVENT',
        'CLASSES_ENABLED': 'CLASSES_ENABLED',
        'CLASS_SCHEDULE_MODE': 'CLASS_SCHEDULE_MODE',
        'CLASSES_PER_PERIOD': 'CLASSES_PER_PERIOD',
        'CLASS_SIZE': 'CLASS_SIZE',
        'CLASS_PRICE': 'CLASS_PRICE',
        'CLASS_CONV_RATE': 'CLASS_CONV_RATE',
        'CLASS_CONV_LAG_MO': 'CLASS_CONV_LAG_MO',
        'BASE_EVENTS_PER_MONTH_LAMBDA': 'BASE_EVENTS_PER_MONTH_LAMBDA',
        'EVENTS_MAX_PER_MONTH': 'EVENTS_MAX_PER_MONTH',
        'TICKET_PRICE': 'TICKET_PRICE',
        'LOAN_504_ANNUAL_RATE': 'LOAN_504_ANNUAL_RATE',
        'LOAN_504_TERM_YEARS': 'LOAN_504_TERM_YEARS',
        'IO_MONTHS_504': 'IO_MONTHS_504',
        'LOAN_7A_ANNUAL_RATE': 'LOAN_7A_ANNUAL_RATE',
        'LOAN_7A_TERM_YEARS': 'LOAN_7A_TERM_YEARS',
        'IO_MONTHS_7A': 'IO_MONTHS_7A',
        'LOAN_CONTINGENCY_PCT': 'LOAN_CONTINGENCY_PCT',
        'RUNWAY_MONTHS': 'RUNWAY_MONTHS',
        'EXTRA_BUFFER': 'EXTRA_BUFFER',
        'RESERVE_FLOOR': 'RESERVE_FLOOR',
        'grant_amount': 'grant_amount',
        'grant_month': 'grant_month',
    }
    
    # Apply mappings
    overrides = {}
    for param_name, value in scenario_params.items():
        if param_name in param_mapping:
            simulator_param_name = param_mapping[param_name]
            overrides[simulator_param_name] = value
        else:
            # Pass through unmapped parameters (like CAPEX_ITEMS, etc.)
            overrides[param_name] = value
    
    return overrides
