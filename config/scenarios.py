#!/usr/bin/env python3
"""
Scenario and strategy definitions for the pottery studio simulator.
Contains all predefined environments and business strategies.
"""

# Predefined scenario environments (external conditions)
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

# Predefined business strategies (internal choices)
STRATEGIES = [
    {"name":"Enhanced_A", "MONTHLY_RENT":4000, "OWNER_COMPENSATION":2000, "MEMBERSHIP_PRICE": 185,
     "CLASS_SCHEDULE_MODE": "semester", "CLASSES_PER_PERIOD": 2},
    {"name":"Enhanced_B", "MONTHLY_RENT":4000, "OWNER_COMPENSATION":2000, "MEMBERSHIP_PRICE": 185,
     "CLASS_SCHEDULE_MODE": "semester", "CLASSES_PER_PERIOD": 2},
]

# Default equipment configurations
DEFAULT_CAPEX_ITEMS = [
    {"enabled": True,  "label": "Kiln #1 Skutt 1227", "count": 1,  "unit_cost": 7000, "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": True,  "label": "Wheels",       "count": 12,  "unit_cost": 3000,  "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": True,  "label": "Wire racks",   "count": 5,  "unit_cost": 150,  "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": True,  "label": "Clay traps",   "count": 1,  "unit_cost": 160,  "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": True, "label": "Kiln #2 Skutt 1427", "count": 1,  "unit_cost": 10000, "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": True, "label": "Wire racks",   "count": 7,  "unit_cost": 150,  "month": 0,   "member_threshold": None, "finance_504": True},
    {"enabled": False, "label": "Wheels",       "count": 10, "unit_cost": 800,  "month": 6,   "member_threshold": None, "finance_504": True},
    {"enabled": True, "label": "Slab roller",  "count": 1,  "unit_cost": 1800, "month": None,"member_threshold": 50, "finance_504": True},
]
