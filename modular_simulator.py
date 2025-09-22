#!/usr/bin/env python
# coding: utf-8




# In[11]:


"""
Ceramics Studio Simulation — Segmented Ramp + WOM + Tenure Churn + Capacity (+ DSCR & Min-Cash)
===============================================================================================

OVERVIEW
- Monthly Monte Carlo sim of a members-only pottery studio.
- Joins come from 3 market pools (community-studio, home-studio, no-access), plus referrals.
- Joins are shaped by seasonality, awareness ramp, word-of-mouth (WOM), price elasticity, capacity damping, and downturns.
- Churn is tenure-based (higher early, lower later) with optional uplift when the studio is crowded.
- We track revenues (membership, clay, firing, events, workshops), costs (fixed/variable), cash, loan, grants.

KEY DIALS TO TUNE
- Acquisition: MARKET_POOLS, POOL_BASE_INTENT, AWARENESS_*, WOM_Q, WOM_SATURATION, REFERRAL_RATE_PER_MEMBER, REFERRAL_CONV,
  PRICE, PRICE_ELASTICITY, DOWNTURN_*.
- Switching windows: CLASS_TERM_MONTHS, CS_UNLOCK_FRACTION_PER_TERM.
- Capacity feel: STATIONS[*]{capacity, alpha, kappa}, CAPACITY_DAMPING_BETA, OPEN_HOURS_PER_WEEK.
- Retention: ARCHETYPE_MONTHLY_CHURN, tenure multipliers (early, steady, late), UTILIZATION_CHURN_UPLIFT.
- Cash survivability: RENT_SCENARIOS, OWNER_DRAW_SCENARIOS, RUNWAY_MONTHS, LOAN_*, grants & timing.

MONTHLY FLOW (t = 0..T-1)
1) Update eligible switchers (community-studio):
   if t>0 and t % CLASS_TERM_MONTHS == 0:
       unlock = floor(P_CS_remaining * CS_UNLOCK_FRACTION_PER_TERM)
       cs_eligible += unlock; P_CS_remaining -= unlock

2) Build common join multiplier:
   F_t = S_t * J_t * A_t * W_t * D_t * ε_t * M_price
   where:
     S_t  = seasonality[t % 12]
     J_t  = DOWNTURN_JOIN_MULT if downturn else 1
     A_t  = linear ramp from AWARENESS_RAMP_START_MULT → AWARENESS_RAMP_END_MULT over AWARENESS_RAMP_MONTHS
     W_t  = 1 + WOM_Q * (N_t / WOM_SATURATION)
     D_t  = max(0, 1 - (N_t / SoftCap) ** CAPACITY_DAMPING_BETA)
     ε_t  ~ LogNormal(mean=-(σ^2)/2, sigma=ADOPTION_SIGMA)  # mean-one noise


3) Organic joins (binomial by pool):
   p_pool = BASE_INTENT_pool * F_t
   J_no_access ~ Binomial(P_no_access_remaining, p_no_access)
   J_home      ~ Binomial(P_home_remaining, p_home)
   J_cs        ~ Binomial(cs_eligible,       p_cs)
   Decrement each pool by its joins.

4) Referrals (small K-factor):
   J_ref ~ Poisson(REFERRAL_RATE_PER_MEMBER * N_t * REFERRAL_CONV)
   Cap by remaining supply; allocate (no-access → cs_eligible → home); decrement those pools.

5) Onboarding cap:
   J_tot = J_no_access + J_home + J_cs + J_ref
   If J_tot > MAX_ONBOARDINGS_PER_MONTH:
       Roll back overflow proportionally across sources and return people to the correct pools.

6) Create members:
   For each join, draw archetype by MEMBER_ARCHETYPES[*]["prob"]; store start_month, monthly_fee, clay_bags.

7) Tenure-based churn (hazard per member):
   base = ARCHETYPE_MONTHLY_CHURN[arch]
   h(tenure) ≈ 1.8*base (months ≤2), base (3–6), 0.7*base (≥7)
   p_leave = h * (DOWNTURN_CHURN_MULT if downturn else 1) * (1 + UTILIZATION_CHURN_UPLIFT * max(0, N_t/SoftCap - 1))
   Keep member if rand() > p_leave.

8) Ops & cash:
   Revenues = membership fees + clay + firing (stepped fee) + events (Poisson with seasonality) + workshops (probabilistic, net).
   Variable costs = clay COGS + water + electricity (firings_this_month(N_t), kiln #2 if on).
   Fixed OpEx = rent + insurance + glaze + seasonal heating.
   Operating profit = Revenues - (Fixed + Variable).
   Cash OpEx = Fixed + Variable + loan_payment + owner_draw (optionally tapered).
   Cash_t+1 = Cash_t + (Revenues - Cash OpEx); add grants on grant_month.
   Member-triggered purchases are driven by CAPEX_ITEMS (e.g., by month or member thresholds).

CAPACITY (soft cap)
SoftCap = min over stations s of:
  (alpha_s * capacity_s * OPEN_HOURS_PER_WEEK) /
  (kappa_s * sum_arch(prob_arch * sessions_per_week_arch * hours_arch * usage_share_arch,s))

FINANCE
Loan principal at t=0 includes CapEx (depending on scenario), contingency, and runway months of burn (rent + OpEx + owner draw).
Monthly loan payment = standard amortization with LOAN_ANNUAL_RATE and LOAN_TERM_YEARS.
We also compute **monthly DSCR** ≈ Operating Profit / Debt Service.

OUTPUTS (per scenario × rent × draw, aggregated over simulations)
- Membership trajectory (median + 10–90%).
- Cash balance bands with grant markers.
- Median months to operating break-even (cumulative op profit ≥ 0).
- % insolvent before grant.
- Median final cash at month T.
- **Median minimum cash** across the horizon (stress indicator).
- **DSCR** summaries.
- Diagnostics: net adds distribution; optional joins-by-source vs departures plot.
"""


# In[19]:

from __future__ import annotations
from typing import Optional  # put this at the top of your script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numpy.random import default_rng, SeedSequence
from datetime import datetime, date
import matplotlib
matplotlib.use("Agg")  # before importing pyplot
import json
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
from collections import OrderedDict
from types import SimpleNamespace
from contextlib import contextmanager
import copy
import inspect
import re

mpl.rcParams['font.family'] = 'Noto Sans'  # or another installed font with U+2011

_MISSING = object()

def _is_constant_name(name: str) -> bool:
    """Heuristic: ALL_CAPS names are treated as config constants."""
    return name.isupper() and not name.startswith("_")

def get_default_cfg() -> dict:
    """
    Snapshot the current module constants (ALL_CAPS) as defaults.
    This captures the values you’ve already defined in this file.
    """
    g = globals()
    out = {}
    for k, v in list(g.items()):
        if _is_constant_name(k):
            # deepcopy so callers can't mutate module state by reference
            try:
                out[k] = copy.deepcopy(v)
            except Exception:
                out[k] = v
    return out

@contextmanager
def override_globals(new_vals: dict):
    """
    Temporarily override module globals with values from new_vals, then restore.
    This lets the existing code keep using bare names (MONTHS, PRICE, …)
    while making the whole run controlled by a cfg dict.
    """
    old = {}
    g = globals()
    try:
        for k, v in new_vals.items():
            old[k] = g.get(k, _MISSING)
            g[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                g.pop(k, None)
            else:
                g[k] = v

def resolve_cfg(user_cfg: dict | None = None) -> dict:
    """
    Merge user overrides onto the current file’s defaults, with light type validation
    so scalar overrides don’t clobber expected dicts.
    """
    defaults = get_default_cfg()
    merged = dict(defaults)
    if user_cfg:
        merged.update(user_cfg)  # shallow merge by design

        # --- Guard against bad scalar overrides for dict-typed knobs ---
        if not isinstance(merged.get("POOL_BASE_INTENT", defaults["POOL_BASE_INTENT"]), dict):
            # drop the bad override and fall back to defaults
            merged["POOL_BASE_INTENT"] = dict(defaults["POOL_BASE_INTENT"])

        if not isinstance(merged.get("MARKET_POOLS_INFLOW", defaults["MARKET_POOLS_INFLOW"]), dict):
            merged["MARKET_POOLS_INFLOW"] = dict(defaults["MARKET_POOLS_INFLOW"])

    return merged

# -- Downturn probability sourcing -------------------------------------------
def _get_downturn_prob(cfg):
    """Return (prob, source). Never raises; clamps to [0,1].
    Precedence: explicit cfg override > live nowcast > fallback 0.15
    """
    # 1) explicit override
    if cfg and "DOWNTURN_PROB_PER_MONTH" in cfg:
        try:
            p = float(cfg["DOWNTURN_PROB_PER_MONTH"])
            return max(0.0, min(1.0, p)), "static(cfg)"
        except Exception:
            pass

    # 2) live nowcast
    try:
        from nowcast import run_nowcast
        nc = run_nowcast(save_csv=False)          # prints the banner once
        p = float(nc.get("prob", 0.15))
        if not (p == p):                           # NaN guard
            p = 0.15
        return max(0.0, min(1.0, p)), "nowcast"
    except Exception as e:
        print(f"[nowcast] error: {e.__class__.__name__}: {e} → using 0.15")
        return 0.15, "fallback(0.15)"

def apply_workshops(stream, cfg, T):
    """Monthly workshops: revenue, variable cost, optional conversion to members."""
    if not bool(cfg.get("WORKSHOPS_ENABLED", False)):
        return

    wpm   = float(cfg.get("WORKSHOPS_PER_MONTH", 0.0))
    avg_n = int(cfg.get("WORKSHOP_AVG_ATTENDANCE", 0))
    fee   = float(cfg.get("WORKSHOP_FEE", 0.0))
    var_c = float(cfg.get("WORKSHOP_COST_PER_EVENT", 0.0))
    conv  = float(cfg.get("WORKSHOP_CONV_RATE", 0.0))
    lag   = int(cfg.get("WORKSHOP_CONV_LAG_MO", 1))

    # Expected counts per month (you can add Poisson noise if desired)
    events_pm = wpm
    attendees_pm = int(round(events_pm * avg_n))
    gross_rev_pm = attendees_pm * fee
    var_cost_pm = events_pm * var_c
    net_rev_pm = gross_rev_pm - var_cost_pm
    conv_joins_pm = int(round(attendees_pm * conv))

    for t in range(T):
        # revenue
        stream["workshop_revenue"][t] += net_rev_pm
        # conversions (bounded by horizon)
        conv_t = min(T-1, t + lag)
        stream["joins_from_workshops"][conv_t] += conv_joins_pm



 


# =============================================================================
# Tunable Parameters
# =============================================================================

# -------------------------------------------------------------------------
# Simulation Settings
# -------------------------------------------------------------------------
MONTHS = 60
N_SIMULATIONS = 100
RANDOM_SEED = 42

# -------------------------------------------------------------------------
# Financing & Loans
# -------------------------------------------------------------------------
RUNWAY_MONTHS = 12
LOAN_CONTINGENCY_PCT = 0.08
EXTRA_BUFFER = 0.0  # cushion for sensitivity / sizing sweeps

# --- Aggregate loan (legacy / fallback) ---
LOAN_ANNUAL_RATE = 0.08
LOAN_TERM_YEARS = 5

# --- Split loans ---
LOAN_504_ANNUAL_RATE = 0.070   # effective blended proxy for SBA 504 (equipment)
LOAN_504_TERM_YEARS  = 5
LOAN_7A_ANNUAL_RATE  = 0.115   # SBA 7(a), 5-year term mid-case
LOAN_7A_TERM_YEARS   = 5

DSCR_CASH_TARGET = 1.25  # put with other constants

# --- Interest-only months (can be overridden from UI/batch) ---
try:
    IO_MONTHS_504
except NameError:
    IO_MONTHS_504 = 0

try:
    IO_MONTHS_7A
except NameError:
    IO_MONTHS_7A = 0
                        
# --- SBA fee defaults (overridable from UI/batch) ---
try: FEES_UPFRONT_PCT_7A
except NameError: FEES_UPFRONT_PCT_7A = 0.03   # 3% example
try: FEES_UPFRONT_PCT_504
except NameError: FEES_UPFRONT_PCT_504 = 0.02  # 2% example
try: FEES_PACKAGING
except NameError: FEES_PACKAGING = 2500.0
try: FEES_CLOSING
except NameError: FEES_CLOSING = 1500.0
try: FINANCE_FEES_7A
except NameError: FINANCE_FEES_7A = True
try: FINANCE_FEES_504
except NameError: FINANCE_FEES_504 = True
# --- Firing fee tier defaults (overridable from UI/batch) ---
try: FIRING_FEE_TIER1_LBS
except NameError: FIRING_FEE_TIER1_LBS = 20
try: FIRING_FEE_TIER2_LBS
except NameError: FIRING_FEE_TIER2_LBS = 40
try: FIRING_FEE_TIER1_RATE
except NameError: FIRING_FEE_TIER1_RATE = 3.0
try: FIRING_FEE_TIER2_RATE
except NameError: FIRING_FEE_TIER2_RATE = 4.0
try: FIRING_FEE_TIER3_RATE
except NameError: FIRING_FEE_TIER3_RATE = 5.0

    
# Owner draw taper
OWNER_DRAW_START_MONTH = 1
OWNER_DRAW_END_MONTH   = 12  # set None for indefinite
# Owner stipend controls
OWNER_STIPEND_MONTHS = 12  # how many months of stipend to reserve & actually pay
# (OPTIONAL) If you want calendar limits too, you can keep using OWNER_DRAW_START_MONTH/END
# and we’ll make END default to OWNER_STIPEND_MONTHS unless you override it.

# Staffing expansion trigger
STAFF_EXPANSION_THRESHOLD = 50
STAFF_COST_PER_MONTH = 2500.0

# -------------------------------------------------------------------------
# Community-studio switch windows & onboarding cap (were undefined)
# -------------------------------------------------------------------------
CLASS_TERM_MONTHS = 3                # e.g., 12-week terms → switch window every 3 months
CS_UNLOCK_FRACTION_PER_TERM = 0.25   # share of remaining CS pool that becomes eligible each window
MAX_ONBOARDINGS_PER_MONTH = None     # set an int (e.g., 6/10/12) to hard-cap monthly new members


# -------------------------------------------------------------------------
# Rent & Draw Scenarios
# -------------------------------------------------------------------------
RENT_SCENARIOS = np.arange(2500, 5001, 1000)
OWNER_DRAW_SCENARIOS = [0, 1000, 2000, 3000]

RENT_GROWTH_PCT = 0.0 # Annual rent escalation, as a fraction (e.g., 0.03 = 3%/yr). Defaults to 0.


# -------------------------------------------------------------------------
# Generalized CapEx schedule (optional).
# Each item may trigger by month, by membership threshold, or either:
#   {"amount": 8000, "month": 6, "label": "Kiln #2"}
#   {"amount": 4000, "member_threshold": 50, "label": "Slab Roller"}
# If CAPEX_ITEMS is empty/undefined, no scheduled purchases are executed.
# -------------------------------------------------------------------------
try:
    CAPEX_ITEMS
except NameError:
    CAPEX_ITEMS = []  # default: no scheduled purchases

# ==== De-Staged loan timing knobs ===========================================
# "upfront": full proceeds at t=0; prebuilt 504/7a schedules apply
# "staged":  draw tranches when purchases execute; payments begin next month
try:
    LOAN_MODE
except NameError:
    LOAN_MODE = "upfront"   # "upfront" | "staged"

# Staged-draw rule (only used if LOAN_MODE == "staged")
try:
    LOAN_STAGED_RULE
except NameError:
    LOAN_STAGED_RULE = {
        "draw_pct_of_purchase": 1.00,   # 100% of this month's CAPEX funded by debt
        "min_tranche": 0.0,             # floor
        "max_tranche": None,            # optional cap
    }

# Optional: how many months to lump together when turning the CapEx table into a schedule
try:
    CAPEX_LUMP_WINDOW_MONTHS
except NameError:
    CAPEX_LUMP_WINDOW_MONTHS = 2  # "a couple of months"

from collections import defaultdict
import numpy as np

def _lump_capex_table(rows, horizon, window_months=2, staged_pct=1.0, label_months=None, default_month=0):
    """
    Turn a raw equipment table into a lumped month-by-month purchase schedule.
    Returns (purchases_ts, eligible_ts, details_by_anchor, anchor_map_per_label).
    """
    if label_months is None:
        label_months = {}
    items = []
    for r in rows:
        if not r.get('enabled', True):
            continue
        qty = int(r.get('count', 0) or 0)
        u   = float(r.get('unit_cost', 0.0) or 0.0)
        if qty <= 0 or u <= 0:
            continue
        m   = int(label_months.get(r.get('label', ''), r.get('month', default_month) or default_month))
        cost = float(qty * u)
        items.append((m, r.get('label', ''), cost))

    wnd = max(1, int(window_months))
    buckets, details, anchors = defaultdict(float), defaultdict(list), {}
    for m, label, cost in items:
        anchor = (int(m) // wnd) * wnd
        anchors[label] = anchor
        buckets[anchor] += cost
        details[anchor].append((label, cost, int(m)))

    purchases = np.zeros(int(horizon), dtype=float)
    for anchor, amt in buckets.items():
        if 0 <= anchor < len(purchases):
            purchases[anchor] += amt

    eligible = staged_pct * purchases
    return purchases, eligible, details, anchors


def _tranche_from_schedule(eligible, min_tranche, max_tranche=None, tail_policy="draw"):
    """
    Bucket accumulate 'eligible' until >= min_tranche; fire capped draws (<= max_tranche).
    Returns monthly draws array.
    """
    n = len(eligible)
    draws = np.zeros(n, dtype=float)
    bucket = 0.0
    cap = float(max_tranche) if (max_tranche not in (None, 0)) else float("inf")
    floor = float(min_tranche or 0.0)

    # If no minimum tranche, draw-as-you-go (respecting cap)
    if floor <= 0:
        e = np.asarray(eligible, dtype=float)
        return np.minimum(e, cap) if np.isfinite(cap) else e.copy()


    for m in range(n):
        bucket += float(eligible[m])
        while bucket + 1e-12 >= floor:   # tiny epsilon prevents float stalls
            take = min(bucket, cap)
            if take <= 0:
                break

    if tail_policy == "draw" and bucket > 0 and n > 0:
        draws[-1] += bucket
        bucket = 0.0
    return draws

# If provided via overrides, this amount replaces computed principals for "upfront" mode.
try:
    LOAN_UPFRONT_PROCEEDS
except NameError:
    LOAN_UPFRONT_PROCEEDS = None  # None = use computed 504+7a sizing

# -------------------------------------------------------------------------
# Operating Expenses (recurring)
# -------------------------------------------------------------------------
INSURANCE_COST = 75
GLAZE_COST_PER_MONTH = 833.33
HEATING_COST_WINTER = 450
HEATING_COST_SUMMER = 30

# Utilities
COST_PER_KWH = 0.2182
KWH_PER_FIRING_KMT1027 = 75
KWH_PER_FIRING_KMT1427 = 110
WATER_COST_PER_GALLON = 0.02
GALLONS_PER_BAG_CLAY = 1
WHOLESALE_CLAY_COST_PER_BAG = 16.75

# -------------------------------------------------------------------------
# Electricity & Kiln Scheduling
# -------------------------------------------------------------------------
DYNAMIC_FIRINGS = True
BASE_FIRINGS_PER_MONTH = 10
REFERENCE_MEMBERS_FOR_BASE_FIRINGS = 12
MIN_FIRINGS_PER_MONTH = 4
MAX_FIRINGS_PER_MONTH = 12

# -------------------------------------------------------------------------
# --- Beginner Classes (opt-in offering) ---
# -------------------------------------------------------------------------

CLASSES_ENABLED = False
CLASS_COHORTS_PER_MONTH = 2
CLASS_CAP_PER_COHORT = 10
CLASS_FILL_MEAN = 0.85          # avg fill percentage (0–1)
CLASS_PRICE = 600             # per student
CLASS_COST_PER_STUDENT = 40.0   # materials/admin
CLASS_INSTR_RATE_PER_HR = 30.0
CLASS_HOURS_PER_COHORT = 3.0
CLASS_CONV_RATE = 0.12          # fraction converting to members
CLASS_CONV_LAG_MO = 1           # months after class end
CLASS_EARLY_CHURN_MULT = 0.8    # first 3–6 months lower churn for converts

# ---- Semester Calendar for Classes ----
# Switch between monthly classes and 4 terms/year.
# "semester" mode: classes only run during defined months.
CLASSES_CALENDAR_MODE = "semester"  # options: "semester", "monthly"
CLASS_SEMESTER_LENGTH_MONTHS = 3    # e.g., 3-month terms
# Start months within a year (0=Jan, 3=Apr, 6=Jul, 9=Oct)
CLASS_SEMESTER_START_MONTHS = [0, 3, 6, 9]

def _is_class_month(month: int) -> bool:
    """Return True if classes run in this month under the configured calendar."""
    if not CLASSES_ENABLED:
        return False
    # Date-based gate: figure out how many months until Jan 1, 2026 from today.
    # For those initial months of the simulation, do not allow classes even if enabled.
    _gate_months = _months_until_date(date.today(), date(2026, 1, 1))
    if month < _gate_months:
        return False


    if CLASSES_CALENDAR_MODE == "monthly":
        return True
    m = month % 12
    for start in CLASS_SEMESTER_START_MONTHS:
        # Within the window [start, start+length)
        if 0 <= (m - start) < CLASS_SEMESTER_LENGTH_MONTHS:
            return True
    return False

# -------------------------------------------------------------------------
# Revenue: Memberships
# -------------------------------------------------------------------------
MEMBER_ARCHETYPES = {
    "Hobbyist":          {"prob": 0.35, "monthly_fee": 175, "clay_bags": (0.25, 0.5, 1)},
    "Committed Artist":  {"prob": 0.40, "monthly_fee": 185, "clay_bags": (1, 1.5, 2)},
    "Production Potter": {"prob": 0.10, "monthly_fee": 200, "clay_bags": (2, 2.5, 3)},
    "Seasonal User":     {"prob": 0.15, "monthly_fee": 150, "clay_bags": (0.25, 0.5, 1)},
}

# -------------------------------------------------------------------------
# Revenue: Events (paint-a-pot / sip-&-paint)
# -------------------------------------------------------------------------
EVENTS_ENABLED = True
EVENTS_MAX_PER_MONTH = 4
EVENT_MUG_COST_RANGE = (4.50, 7.50)   # Bisque Imports stoneware mugs
EVENT_CONSUMABLES_PER_PERSON = 2.50   # glaze, brushes, wipes, packaging
EVENT_STAFF_RATE_PER_HOUR = 22.0       # set >0 to include staff costs
EVENT_HOURS_PER_EVENT = 2.0
ATTENDEES_PER_EVENT_RANGE = [8, 10, 12]
TICKET_PRICE = 75

# Seasonality
SEASONALITY_WEIGHTS = np.array([1.1, 1.2, 1.3, 1.4, 1.3, 0.9, 0.8, 0.85, 1.3, 1.4, 1.2, 1.0])
NORMALIZE_SEASONALITY = True
BASE_EVENTS_PER_MONTH_LAMBDA = 3
SEASONALITY_WEIGHTS_NORM = (
    SEASONALITY_WEIGHTS / SEASONALITY_WEIGHTS.mean()
    if NORMALIZE_SEASONALITY else SEASONALITY_WEIGHTS
)

# -------------------------------------------------------------------------
# Revenue: Add-ons
# -------------------------------------------------------------------------
RETAIL_CLAY_PRICE_PER_BAG = 25

# Designated Studios
DESIGNATED_STUDIO_COUNT = 2
DESIGNATED_STUDIO_PRICE = 300.0
DESIGNATED_STUDIO_BASE_OCCUPANCY = 0.3

# -------------------------------------------------------------------------
# Membership Dynamics (churn, adoption, capacity)
# -------------------------------------------------------------------------
ARCHETYPE_MONTHLY_CHURN = {
    "Hobbyist":          0.049 * 0.95,
    "Committed Artist":  0.049 * 0.80,
    "Production Potter": 0.049 * 0.65,
    "Seasonal User":     0.049 * 1.90,
}
MIN_STAY = 1
MAX_STAY = 48

# Downturn regime
DOWNTURN_JOIN_MULT  = 0.65
DOWNTURN_CHURN_MULT = 1.50

# Market pools
MARKET_POOLS = {"community_studio": 70, "home_studio": 50, "no_access": 20}
MARKET_POOLS_INFLOW = {"community_studio": 4, "home_studio": 2, "no_access": 3}
POOL_BASE_INTENT = {"community_studio": 0.10, "home_studio": 0.010, "no_access": 0.040}

# Word-of-mouth
WOM_Q = 0.60
WOM_SATURATION = 60
ADOPTION_SIGMA = 0.20
AWARENESS_RAMP_MONTHS = 4
AWARENESS_RAMP_START_MULT = 0.5
AWARENESS_RAMP_END_MULT = 1.0

# ==== Compartment join model (first principles) ====
# Toggle: "baseline" (current behavior) or "compartment" (first principles)
JOIN_MODEL = str(globals().get("JOIN_MODEL", "compartment")).lower()

# Initial pool stocks (people not yet members) — defaults 0 to respect zero-boundary
NO_ACCESS_POOL = int(globals().get("NO_ACCESS_POOL", 0))
HOME_POOL      = int(globals().get("HOME_POOL", 0))
COMMUNITY_POOL = int(globals().get("COMMUNITY_POOL", 0))

# Monthly inflow (new people entering each pool) — defaults 0
NO_ACCESS_INFLOW   = int(globals().get("NO_ACCESS_INFLOW", 0))
HOME_INFLOW        = int(globals().get("HOME_INFLOW", 0))
COMMUNITY_INFLOW   = int(globals().get("COMMUNITY_INFLOW", 0))

# Per-segment baseline join hazards (per capita per month) — defaults 0
BASELINE_RATE_NO_ACCESS = float(globals().get("BASELINE_RATE_NO_ACCESS", 0.0))
BASELINE_RATE_HOME      = float(globals().get("BASELINE_RATE_HOME", 0.0))
BASELINE_RATE_COMMUNITY = float(globals().get("BASELINE_RATE_COMMUNITY", 0.0))

def _haz_to_prob(lam: float) -> float:
    """Convert monthly hazard to probability in [0,1]."""
    if lam <= 0.0:
        return 0.0
    return float(1.0 - np.exp(-float(lam)))

# Capacity & utilization
OPEN_HOURS_PER_WEEK = 16 * 7  # 112 hours
STATIONS = {
    "wheels":       {"capacity": 8, "alpha": 0.80, "kappa": 2},
    "handbuilding": {"capacity": 6, "alpha": 0.50, "kappa": 3.0},
    "glaze":        {"capacity": 6, "alpha": 0.55, "kappa": 2.6},
}
SESSIONS_PER_WEEK = {"Hobbyist": 1.0, "Committed Artist": 1.5,
                     "Production Potter": 3.5, "Seasonal User": 0.75}
SESSION_HOURS = {"Hobbyist": 1.7, "Committed Artist": 2.75,
                 "Production Potter": 3.8, "Seasonal User": 2.0}
USAGE_SHARE = {
    "Hobbyist": {"wheels": 0.50, "handbuilding": 0.35, "glaze": 0.15},
    "Committed Artist": {"wheels": 0.45, "handbuilding": 0.35, "glaze": 0.20},
    "Production Potter": {"wheels": 0.60, "handbuilding": 0.25, "glaze": 0.15},
    "Seasonal User": {"wheels": 0.40, "handbuilding": 0.45, "glaze": 0.15},
}

CAPACITY_DAMPING_BETA = 4

# -------------------------------------------------------------------------
# Pricing & Referrals
# -------------------------------------------------------------------------
PRICE = 175
JOIN_PRICE_ELASTICITY = -0.6
CHURN_PRICE_ELASTICITY = 0.3
BASELINE_JOIN_RATE = 0.013
REFERRAL_RATE_PER_MEMBER = 0.06
REFERRAL_CONV = 0.22
MAX_MEMBERS = 77
# --- Equipment defaults (overridable via cfg) ---
UTILIZATION_CHURN_UPLIFT = 0.25

# -------------------------------------------------------------------------
# Taxation & Entity Setup (Massachusetts)
# -------------------------------------------------------------------------
ENTITY_TYPE = "sole_prop"

# Individual / SE
MA_PERSONAL_INCOME_TAX_RATE = 0.05
SE_EARNINGS_FACTOR = 0.9235
SE_SOC_SEC_RATE = 0.124
SE_MEDICARE_RATE = 0.029
SE_SOC_SEC_WAGE_BASE = 168_600

# Payroll (S-corp)
SCORP_OWNER_SALARY_PER_MONTH = 4000.0
EMPLOYEE_PAYROLL_TAX_RATE = 0.0765
EMPLOYER_PAYROLL_TAX_RATE = 0.0765

# Corporate (C-corp)
FED_CORP_TAX_RATE = 0.21
MA_CORP_TAX_RATE  = 0.08

# Sales & property taxes
MA_SALES_TAX_RATE = 0.0625
PERSONAL_PROPERTY_TAX_ANNUAL = 0.0
PERSONAL_PROPERTY_TAX_BILL_MONTH = 3

# Remittance cadence
ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS = 3
SALES_TAX_REMIT_FREQUENCY_MONTHS = 3

# -------------------------------------------------------------------------
# Maintenance & Marketing
# -------------------------------------------------------------------------
MAINTENANCE_BASE_COST = 200.0
MAINTENANCE_RANDOM_STD = 150.0

MARKETING_COST_BASE = 300.0
MARKETING_RAMP_MONTHS = 12
MARKETING_RAMP_MULTIPLIER = 2.0

# -------------------------------------------------------------------------
# Scenarios (with grants)
# -------------------------------------------------------------------------
SCENARIO_CONFIGS = [
    {"name": "Base", "grant_amount": 0.0, "grant_month": None},
]


def _to_serializable(x):
    try:
        import numpy as np
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    return x

# ---- EFFECTIVE CONFIG ECHO (place near the end, after overrides/globals are set) ----

def _g(name, default=None):
    return globals().get(name, default)

def _ser(x):
    # ensure numpy / arrays are JSONable
    try:
        return _to_serializable(x)
    except Exception:
        return x

EFFECTIVE_CONFIG = OrderedDict({
    "MONTHS": _g("MONTHS"),
    "N_SIMULATIONS": _g("N_SIMULATIONS"),
    "PRICE": _g("PRICE"),
    "JOIN_PRICE_ELASTICITY": _g("JOIN_PRICE_ELASTICITY"),
    "CHURN_PRICE_ELASTICITY": _g("CHURN_PRICE_ELASTICITY"),
    "RENT_SCENARIOS": _ser(_g("RENT_SCENARIOS")),
    "OWNER_DRAW_SCENARIOS": _ser(_g("OWNER_DRAW_SCENARIOS")),
    "DOWNTURN_PROB_PER_MONTH": _g("DOWNTURN_PROB_PER_MONTH"),
    "DOWNTURN_JOIN_MULT": _g("DOWNTURN_JOIN_MULT"),
    "DOWNTURN_CHURN_MULT": _g("DOWNTURN_CHURN_MULT"),
    "MARKET_POOLS_INFLOW": _ser(_g("MARKET_POOLS_INFLOW")),
    "WOM_Q": _g("WOM_Q"),
    "AWARENESS_RAMP_MONTHS": _g("AWARENESS_RAMP_MONTHS"),
    "HARD_CAP": _g("HARD_CAP"),
    "CAPACITY_SOFT_CAP": _g("CAPACITY_SOFT_CAP"),
})

# Write once, after fully assembled
try:
    out_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_effective")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "effective_config.json", "w") as f:
        json.dump(EFFECTIVE_CONFIG, f, indent=2, default=_to_serializable)
except Exception:
    pass

print("[EFFECTIVE_CONFIG]", json.dumps(EFFECTIVE_CONFIG, default=_to_serializable))

# =============================================================================
# Helpers
# =============================================================================


def _months_until_date(start: date, target: date) -> int:
    """
    Number of whole simulation months to reach `target` from `start`.
    Partial current months do not count as a full month.
    Example: from Aug 27, 2025 to Jan 1, 2026 -> 4 (Sep, Oct, Nov, Dec).
    """
    if start >= target:
        return 0
    months = (target.year - start.year) * 12 + (target.month - start.month)
    # If we're already partway through the current month, don't count it as a full month.
    if start.day > 1:
        months -= 1
    return max(0, months)


def seasonal_churn_mult(month_idx: int) -> float:
    """
    Multiplier for churn probability based on month of year.
    month_idx is 0-based in the sim; convert to human month (1–12).
    """
    m = (month_idx % 12) + 1  # 1..12

    if m in (6, 7, 8):       # Jun–Aug: travel/moves
        return 1.25          # +25% churn
    elif m in (1, 2):        # Jan–Feb: post-holiday dropouts
        return 1.15
    elif m in (11, 12):      # Nov–Dec: holidays, gift memberships expiring
        return 1.10
    else:
        return 1.0
    
    
def in_owner_draw_window(month_idx: int) -> bool:
    """
    Returns True if owner draw should be paid in the given 0-based month index.
    Interprets OWNER_DRAW_START_MONTH/OWNER_DRAW_END_MONTH as 1-based, inclusive.
    If OWNER_DRAW_END_MONTH is None, the window is [START, ∞).
    """
    m = month_idx + 1  # convert to 1-based for human-friendly comparison
    start = int(OWNER_DRAW_START_MONTH)
    end = OWNER_DRAW_END_MONTH
    if end is None:
        return m >= start
    return start <= m <= int(end)


def sample_capex(capex_dict, rng):
    return sum(rng.triangular(low, mode, high) for (low, mode, high) in capex_dict.values())


def draw_adopters(remaining_pool, monthly_intent, rng):
    """
    Stochastic adoption draw from a pool with intent rate.
    We use a Binomial draw clipped by remaining_pool for realism.
    """
    intent = np.clip(monthly_intent, 0.0, 1.0)
    return int(rng.binomial(n=int(max(0, remaining_pool)), p=float(intent)))

def calculate_monthly_payment(principal, annual_rate, years):
    if annual_rate == 0:
        return principal / (years * 12)
    r = annual_rate / 12
    n = years * 12
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def build_loan_schedule(principal: float, annual_rate: float, term_years: int,
                        io_months: int, total_months: int) -> np.ndarray:
    """
    Return a length-`total_months` array of monthly payments.
    First `io_months`: interest-only; thereafter level amortization over remaining term months.
    Payments beyond the loan term are zeros.
    """
    if principal <= 0 or total_months <= 0:
        return np.zeros(max(int(total_months), 0), dtype=float)

    r = float(annual_rate) / 12.0
    term_m = int(term_years * 12)
    io_m = int(max(0, io_months))
    io_m = min(io_m, term_m)  # IO cannot exceed term

    pays = np.zeros(int(total_months), dtype=float)

    # Interest-only phase
    io_payment = principal * r if r != 0.0 else 0.0
    io_len = min(io_m, total_months)
    if io_len > 0:
        pays[:io_len] = io_payment

    # Amortization phase
    rem_term = term_m - io_len
    if rem_term > 0 and io_len < total_months:
        if r == 0.0:
            amort_payment = principal / rem_term
        else:
            amort_payment = principal * (r / (1.0 - (1.0 + r) ** (-rem_term)))
        pays[io_len: min(total_months, io_len + rem_term)] = amort_payment

    # Beyond loan maturity: zeros
    return pays

# --- Firing fee schedule defaults (overridable) ---
try:
    FIRING_FEE_SCHEDULE
except NameError:
    # List of tiers: each item is {"up_to_lbs": int|None, "rate": float}
    # None for "no upper limit" final tier
    FIRING_FEE_SCHEDULE = [
        {"up_to_lbs": 20, "rate": 3.0},
        {"up_to_lbs": 40, "rate": 4.0},
        {"up_to_lbs": None, "rate": 5.0},
    ]

# --- Manual membership curve override (UI/adapter can set these) ---
try:
    USE_MANUAL_MEMBERSHIP_CURVE
except NameError:
    USE_MANUAL_MEMBERSHIP_CURVE = False
try:
    MANUAL_MEMBERSHIP_CURVE
except NameError:
    MANUAL_MEMBERSHIP_CURVE = []

def compute_firing_fee(clay_lbs):
    """
    Compute firing fee revenue from lbs, using overridable tier schedule.
    Accepts FIRING_FEE_SCHEDULE as a list of {"up_to_lbs": int|None, "rate": float}
    or as a JSON string with the same structure.
    """
    sched = globals().get("FIRING_FEE_SCHEDULE")
    # Allow JSON string override
    if isinstance(sched, str):
        try:
            import json as _json
            sched = _json.loads(sched)
        except Exception:
            sched = None
    # Fallback if override missing/bad
    if not isinstance(sched, list) or not sched:
        sched = [
            {"up_to_lbs": 20, "rate": 3.0},
            {"up_to_lbs": 40, "rate": 4.0},
            {"up_to_lbs": None, "rate": 5.0},
        ]
    # Enforce ascending order and sane values
    def _upper(x):
        return float("inf") if x is None else float(x)
    total = 0.0
    remaining = float(clay_lbs)
    prev_cut = 0.0
    for tier in sched:
        up_to = _upper(tier.get("up_to_lbs"))
        rate = float(tier.get("rate", 0.0))
        band = max(0.0, min(remaining, up_to - prev_cut))
        if band > 0:
            total += band * rate
            remaining -= band
        prev_cut = up_to
        if remaining <= 1e-9:
            break
    # If still remaining (no open-ended final tier), charge last rate
    if remaining > 1e-9:
        last_rate = float(sched[-1].get("rate", 0.0))
        total += remaining * last_rate
    return total


def _add_staged_tranche_into_array(arr: np.ndarray, start_month: int, principal: float,
                                   annual_rate: float, amort_years: int, io_months: int, total_months: int):
    """
    Build a per-month payment array for a new tranche and add it into `arr`,
    with payments beginning the month *after* the draw (start_month + 1).
    """
    if principal <= 0:
        return
    # Build schedule relative to month 0
    sched = build_loan_schedule(principal, annual_rate, amort_years, io_months, total_months)
    # Shift by +1 so first payment is next month
    start = min(total_months, max(0, start_month + 1))
    end   = min(total_months, start + len(sched))
    seg_len = end - start
    if seg_len > 0:
        arr[start:end] += sched[:seg_len]




def firings_this_month(n_active_members):
    if not DYNAMIC_FIRINGS:
        return BASE_FIRINGS_PER_MONTH
    raw = BASE_FIRINGS_PER_MONTH * (n_active_members / max(1, REFERENCE_MEMBERS_FOR_BASE_FIRINGS))
    return int(np.clip(round(raw), MIN_FIRINGS_PER_MONTH, MAX_FIRINGS_PER_MONTH))

def compute_membership_soft_cap():
    """Compute soft cap from station capacities and member usage assumptions.
    Defensive against scalar overrides for dict-typed knobs."""
    H = OPEN_HOURS_PER_WEEK

    # --- Defensive coercions for dict-typed configs ---
    _sessions = SESSIONS_PER_WEEK
    if not isinstance(_sessions, dict):
        try:
            v = float(_sessions)
            _sessions = {a: v for a in MEMBER_ARCHETYPES.keys()}
        except Exception:
            _sessions = {"Hobbyist": 1, "Artist": 3, "Pro": 5, "Seasonal": 2}

    _dur = SESSION_HOURS
    if not isinstance(_dur, dict):
        try:
            v = float(_dur)
            _dur = {a: v for a in MEMBER_ARCHETYPES.keys()}
        except Exception:
            _dur = {"Hobbyist": 2.0, "Artist": 3.0, "Pro": 4.0, "Seasonal": 2.0}

    _share = USAGE_SHARE
    if not isinstance(_share, dict):
        _share = {a: {s: 1.0 for s in STATIONS.keys()} for a in MEMBER_ARCHETYPES.keys()}

    caps = {}
    for s, cfg in STATIONS.items():
        denom = 0.0
        for arch, arch_cfg in MEMBER_ARCHETYPES.items():
            mix = arch_cfg["prob"]
            s_per_wk = _sessions[arch]
            dur = _dur[arch]
            share = _share[arch][s]
            denom += mix * s_per_wk * dur * share
        caps[s] = (cfg["alpha"] * cfg["capacity"] * H) / (cfg["kappa"] * denom)
    return min(caps.values()), caps

def awareness_multiplier(month_idx):
    """Smooth ramp from START to END over AWARENESS_RAMP_MONTHS."""
    if AWARENESS_RAMP_MONTHS <= 0: 
        return AWARENESS_RAMP_END_MULT
    t = min(1.0, month_idx / AWARENESS_RAMP_MONTHS)
    return AWARENESS_RAMP_START_MULT + t * (AWARENESS_RAMP_END_MULT - AWARENESS_RAMP_START_MULT)

def wom_multiplier(current_members):
    """Simple Bass-style imitation term, saturating with membership level."""
    if WOM_SATURATION <= 0:
        return 1.0
    # 1 + q * (adopters / K); bounded >= 1
    return 1.0 + WOM_Q * (current_members / WOM_SATURATION)

def compute_cs_unlock_share(month_idx, remaining_cs_pool):
    """
    Every CLASS_TERM_MONTHS, let a fraction of the *remaining* community-studio pool
    become eligible to switch.
    """
    if CLASS_TERM_MONTHS <= 0 or remaining_cs_pool <= 0:
        return 0
    return (
    int(np.floor(remaining_cs_pool * CS_UNLOCK_FRACTION_PER_TERM))
    if ((month_idx % CLASS_TERM_MONTHS) == 0 and (month_idx > 0))
    else 0
    )

def month_churn_prob(arch, tenure_mo):
    """Piecewise hazard: higher early churn, stickier later."""
    base = ARCHETYPE_MONTHLY_CHURN[arch]
    if tenure_mo <= 2:
        return min(0.99, base * 1.8)   # onboarding risk
    if tenure_mo <= 6:
        return base                    # steady state
    return base * 0.7                  # long-stay sticky


# =============================================================================
# Simulation
# =============================================================================
def _core_simulation_and_reports():
    """
    The original script body goes here, unmodified:
    - MEMBERSHIP_SOFT_CAP, PER_STATION_CAPS = compute_membership_soft_cap()
    - the loops over RENT_SCENARIOS, OWNER_DRAW_SCENARIOS, SCENARIO_CONFIGS…
    - building rows -> results_df
    - plots
    - summary_table / owner_takehome_table
    - exports
    Return anything you want programmatic access to.
    """

    MEMBERSHIP_SOFT_CAP, PER_STATION_CAPS = compute_membership_soft_cap()
    print(f"Soft membership cap (multi-station bottleneck): ~{MEMBERSHIP_SOFT_CAP:.1f} members")
    for s, cap in PER_STATION_CAPS.items():
        print(f"  Station cap via {s:12s}: ~{cap:.1f}")
    
    rows = []
    
    for fixed_rent in RENT_SCENARIOS:
        for owner_draw in OWNER_DRAW_SCENARIOS:
            for scen_cfg in SCENARIO_CONFIGS:
                scen_name = scen_cfg["name"]
                
                # Deterministic, per-path RNG: (seed, rent, draw, scenario_index, sim)
                scen_index = next(i for i, s in enumerate(SCENARIO_CONFIGS) if s["name"] == scen_name)            
                
                # --- Snapshot baseline mutable globals so each simulation starts identical ---
                _BASE_STATIONS    = copy.deepcopy(STATIONS)
                _BASE_MAX_MEMBERS = int(MAX_MEMBERS)
                _BASE_CLAY_COGS   = float(globals().get("CLAY_COGS_MULT", 1.0))
                _BASE_PUG_MAINT   = float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0))
                _BASE_SLAB_MAINT  = float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))
                
                
                for sim in range(N_SIMULATIONS):
                    ss = SeedSequence([RANDOM_SEED, int(fixed_rent), int(owner_draw), int(scen_index), int(sim)])
                    rng = default_rng(ss)
                    # --- Reset mutable globals to the baseline for reproducibility ---
                    if isinstance(STATIONS, dict):
                        STATIONS.clear()
                        STATIONS.update(copy.deepcopy(_BASE_STATIONS))
                    else:
                        globals()["STATIONS"] = copy.deepcopy(_BASE_STATIONS)
                    globals()["MAX_MEMBERS"] = _BASE_MAX_MEMBERS
                    globals()["CLAY_COGS_MULT"] = _BASE_CLAY_COGS
                    globals()["PUGMILL_MAINT_COST_PER_MONTH"] = _BASE_PUG_MAINT
                    globals()["SLAB_ROLLER_MAINT_COST_PER_MONTH"] = _BASE_SLAB_MAINT
                                        
                    
                    # CapEx
                    capex_I_cost = 0.0
                    capex_II_cost = 0.0
    
                    # Runway
                    avg_monthly_heat = (HEATING_COST_WINTER + HEATING_COST_SUMMER) / 2
                    runway_costs = (INSURANCE_COST + GLAZE_COST_PER_MONTH + avg_monthly_heat + fixed_rent + owner_draw) * RUNWAY_MONTHS
    
                     # Table-driven CapEx total (unit_cost × count, or amount fallback)
                    capex_table_total = 0.0
                    for _it in CAPEX_ITEMS:
                        if _it.get("enabled") is False:
                            continue
                        unit = float(_it.get("unit_cost", 0.0) or 0.0)
                        cnt  = int(_it.get("count", 1) or 1)
                        total_cost = unit * max(1, cnt)
                        if total_cost <= 0:
                            amt = _it.get("amount")
                            if amt is None:
                                continue
                            total_cost = float(amt)
                        capex_table_total += total_cost
    
    
                    # Loan principal sizing
                   # ----- Split loan sizing: 504 (CapEx) + 7(a) (runway/draw/buffer) -----
                    total_capex_for_loan = capex_table_total

                    loan_504_principal = total_capex_for_loan * (1 + LOAN_CONTINGENCY_PCT)
                    loan_7a_principal  = runway_costs + EXTRA_BUFFER

                    # ---- SBA fees (can be financed into principal or paid in cash) ----
                    fees_7a_pct  = float(FEES_UPFRONT_PCT_7A)  * float(loan_7a_principal)
                    fees_504_pct = float(FEES_UPFRONT_PCT_504) * float(loan_504_principal)
                    flat_fees    = float(FEES_PACKAGING) + float(FEES_CLOSING)  # attach to 7(a) for simplicity
                    fees_7a_total  = fees_7a_pct  + flat_fees
                    fees_504_total = fees_504_pct

                    fees_cash_outflow = 0.0
                    if FINANCE_FEES_7A:
                        loan_7a_principal += fees_7a_total
                    else:
                        fees_cash_outflow += fees_7a_total
                    if FINANCE_FEES_504:
                        loan_504_principal += fees_504_total
                    else:
                        fees_cash_outflow += fees_504_total
                    # ---- end SBA fees ----
                    # Independent modes
                    CAPEX_MODE = str(globals().get("CAPEX_LOAN_MODE", "upfront")).lower()
                    OPEX_MODE  = str(globals().get("OPEX_LOAN_MODE",  "upfront")).lower()
            
                    # Initialize schedules
                    loan_payment_504_ts = np.zeros(MONTHS, dtype=float)
                    loan_payment_7a_ts  = np.zeros(MONTHS, dtype=float)
                    
                    # Helper: balance series consistent with payment schedule
                    def _balance_series(principal, payments, apr, io_m, term_years, months):
                        if principal <= 0:
                            return np.zeros(months, dtype=float)
                        r_m = float(apr) / 12.0
                        term_m = int(12 * int(term_years))
                        bal = float(principal)
                        out = np.zeros(months, dtype=float)
                        for m in range(months):
                            if m >= term_m:
                                bal = 0.0
                            else:
                                interest = bal * r_m
                                principal_paid = 0.0 if m < int(io_m) else max(0.0, float(payments[m]) - interest)
                                bal = max(0.0, bal - principal_paid)
                            out[m] = bal
                        return out
            
                    # Upfront principals (overrides from UI)
                    ov504 = globals().get("LOAN_OVERRIDE_504", None)
                    ov7a  = globals().get("LOAN_OVERRIDE_7A",  None)
                    # Only override if a value was actually provided; otherwise keep the sized amounts
                    if ov504 is not None:
                        loan_504_principal = float(ov504)
                    if ov7a is not None:
                        loan_7a_principal = float(ov7a)
            
                    # Build fixed amortization only for the modes that are upfront
                    if CAPEX_MODE == "upfront" and loan_504_principal > 0:
                        loan_payment_504_ts = build_loan_schedule(
                            loan_504_principal, LOAN_504_ANNUAL_RATE, LOAN_504_TERM_YEARS, IO_MONTHS_504, MONTHS
                        )
                    if OPEX_MODE == "upfront" and loan_7a_principal > 0:
                        loan_payment_7a_ts = build_loan_schedule(
                            loan_7a_principal, LOAN_7A_ANNUAL_RATE, LOAN_7A_TERM_YEARS, IO_MONTHS_7A, MONTHS
                        )
                    loan_payment_total_ts = loan_payment_504_ts + loan_payment_7a_ts
                    # --- Staged OpEx facility state (if enabled) ---
                    opex_rule = globals().get("LOAN_STAGED_RULE_OPEX", {}) or {}
                    opex_facility_limit = float(opex_rule.get("facility_limit", 0.0) or 0.0)
                    opex_remaining = opex_facility_limit            


                    # (scalar monthly_loan_payment_* replaced by per-month schedules)
                                        # Corresponding balances (upfront modes)
                    loan_balance_504_ts = _balance_series(
                        loan_504_principal, loan_payment_504_ts, LOAN_504_ANNUAL_RATE, IO_MONTHS_504, LOAN_504_TERM_YEARS, MONTHS
                    )
                    loan_balance_7a_ts = _balance_series(
                        loan_7a_principal, loan_payment_7a_ts, LOAN_7A_ANNUAL_RATE, IO_MONTHS_7A, LOAN_7A_TERM_YEARS, MONTHS
                    )
                    loan_principal_total = loan_504_principal + loan_7a_principal
                    sized_runway_costs = runway_costs  # keep for reporting
                    
                     # --- Running loan balances for correct staged tracking ---
                    loan_balance_504_ts = np.zeros(MONTHS, dtype=float)
                    loan_balance_7a_ts  = np.zeros(MONTHS, dtype=float)
                    _bal504 = 0.0
                    _bal7a  = 0.0
                    if CAPEX_MODE == "upfront" and loan_504_principal > 0:
                        _bal504 = float(loan_504_principal)
                        loan_balance_504_ts[0] = _bal504
                    if OPEX_MODE == "upfront" and loan_7a_principal > 0:
                        _bal7a = float(loan_7a_principal)
                        loan_balance_7a_ts[0] = _bal7a
                    _r504 = float(LOAN_504_ANNUAL_RATE) / 12.0
                    _r7a  = float(LOAN_7A_ANNUAL_RATE)  / 12.0
    
                    # Tracking
                    cash_balance = 0.0
                    cumulative_op_profit = 0.0
                    cumulative_after_capex = 0.0
                    expansion_triggered = False
                    active_members = []
                    # Dynamic equipment state: count kilns purchased so far
                    _dyn_KILN_COUNT = 0
                    
                    # Generalized CapEx trigger state (copy globals → per-sim queue)
                    _capex_queue = []
                    try:
                        for _it in CAPEX_ITEMS:
                            # honor enabled flag if present (older presets may omit it)
                            if _it.get("enabled") is False:
                                continue
                            unit = float(_it.get("unit_cost", 0.0) or 0.0)
                            cnt  = int(_it.get("count", 1) or 1)
                            mth  = _it.get("month", None)
                            thr  = _it.get("member_threshold", None)
                            lbl  = _it.get("label", "")
                            total_cost = (unit * max(1, cnt))
                            if total_cost <= 0:
                                total_cost = float(_it.get("amount", 0.0) or 0.0)
                            if total_cost > 0 and (mth is not None or thr is not None):
                                _capex_queue.append({
                                    "unit_cost": unit,
                                    "count": cnt,
                                    "month": mth,
                                    "member_threshold": thr,
                                    "label": lbl,
                                    "purchased": False,
                                    "enabled": True,
                                })
                    except Exception:
                        _capex_queue = []
                    
                    # --- Convert the raw CapEx table into a lumped purchasing schedule ---
                    _capex_rows_for_lumping = []
                    for _it in _capex_queue:
                        if _it.get("month") is None:
                            continue  # member-threshold-driven items handled separately
                        _capex_rows_for_lumping.append({
                            "enabled": True,
                            "label": _it.get("label", ""),
                            "count": int(_it.get("count", 1) or 1),
                            "unit_cost": float(_it.get("unit_cost", 0.0) or 0.0),
                            "month": int(_it.get("month", 0) or 0),
                        })
                    
                    _lump_window = int(globals().get("CAPEX_LUMP_WINDOW_MONTHS", 2))
                    _draw_rule   = globals().get("LOAN_STAGED_RULE", {}) or {}
                    _draw_pct    = float(_draw_rule.get("draw_pct_of_purchase", 1.0))
                    _min_tranche = float(_draw_rule.get("min_tranche", 0.0) or 0.0)
                    _max_tranche = _draw_rule.get("max_tranche", None)
                    
                    # Build monthly series
                    _capex_purchases_ts, _capex_eligible_ts, _capex_dbg, _capex_anchor_map = _lump_capex_table(
                        _capex_rows_for_lumping,
                        horizon=MONTHS,
                        window_months=_lump_window,
                        staged_pct=_draw_pct,
                    )
                    
                    # Precompute staged draws from the eligible series
                    if CAPEX_MODE == "staged":
                         _capex_draws_ts = _tranche_from_schedule(
                             _capex_eligible_ts,
                             min_tranche=_min_tranche,
                             max_tranche=_max_tranche,
                             tail_policy="draw",
                         )
                    else:
                         _capex_draws_ts = np.zeros_like(_capex_eligible_ts)
                                        
                    # Anchor each item’s month to its lumped bucket for consistent side-effects
                    for _it in _capex_queue:
                        if _it.get("month") is None:
                            continue
                        lbl = _it.get("label", "")
                        if lbl in _capex_anchor_map:
                            _it["month"] = int(_capex_anchor_map[lbl])

                     # --- Market pool state for this simulation ---

                    # ---- Dynamic equipment effects state (mutable during the run) ----
                    # Start from current globals so effects are incremental and visible to existing functions.
                    _dyn_STATIONS = {k: dict(v) for k, v in STATIONS.items()}
                    _dyn_MAX_MEMBERS = int(MAX_MEMBERS)
                    _dyn_CLAY_COGS_MULT = float(globals().get("CLAY_COGS_MULT", 1.0))
                    _dyn_PUGMILL_MAINT = float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0))
                    _dyn_SLAB_MAINT = float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))

                    remaining_pool = {
                        "community_studio": int(COMMUNITY_POOL),
                        "home_studio":      int(HOME_POOL),
                        "no_access":        int(NO_ACCESS_POOL),
                    }
    
                    # Community-studio: track an "eligible to switch" sub-pool.
                    cs_eligible = 0
    
                    insolvent_before_grant = False
                    grant_month = scen_cfg["grant_month"]
                    grant_amount = scen_cfg["grant_amount"]
                    
                    # >>> BEGIN classes: per-simulation state
                    pending_class_conversions = {}   # {target_month: count}
                    # >>> END classes
                    
                    # >>> BEGIN workshops: per-simulation state
                    stream = {}
                    stream["workshop_revenue"] = np.zeros(MONTHS)
                    stream["joins_from_workshops"] = np.zeros(MONTHS, dtype=int)
                    # Precompute monthly workshops using UI-configured knobs
                    if bool(globals().get("WORKSHOPS_ENABLED", False)):
                            apply_workshops(stream, globals(), MONTHS)
                    
                    # >>> END workshops
                    
                    # --- Tax/state trackers (reset each simulation) ---
                    se_ss_wage_base_used_ytd = 0.0        # for SE Social Security cap (sole/partnership)
                    se_tax_payable_accum = 0.0            # accrued SE tax (sole/partnership)
                    state_tax_payable_accum = 0.0         # accrued MA personal income tax (pass-through)
                    corp_tax_payable_accum = 0.0          # accrued corporate income tax (C-corp)
                    sales_tax_payable_accum = 0.0         # accrued sales tax to remit
                    tax_payments_this_month = 0.0         # cash paid this month for taxes
    
                    # >>> BEGIN events: capture config-driven knobs once per simulation
                    _g = globals()
                    events_enabled       = bool(_g.get("EVENTS_ENABLED", True))
                    events_max_per_month = int(_g.get("EVENTS_MAX_PER_MONTH", 4))
                    base_lambda          = float(_g.get("BASE_EVENTS_PER_MONTH_LAMBDA", 3.0))
                    ticket_price         = float(_g.get("TICKET_PRICE", 75.0))
                    attendees_range      = list(_g.get("ATTENDEES_PER_EVENT_RANGE", [8, 10, 12]))
                    mug_cost_range       = tuple(_g.get("EVENT_MUG_COST_RANGE", (4.5, 7.5)))
                    consumables_pp       = float(_g.get("EVENT_CONSUMABLES_PER_PERSON", 2.5))
                    staff_rate_hr        = float(_g.get("EVENT_STAFF_RATE_PER_HOUR", 22.0))
                    hours_per_event      = float(_g.get("EVENT_HOURS_PER_EVENT", 2.0))
                    # >>> END events
                    
                    _g = globals()
                    
                    # --- Pricing elasticity setup (fixed baseline) ---
                    price = float(globals().get("PRICE", 165.0))
                    reference_price = float(globals().get("REFERENCE_PRICE", price))  # default to current price if missing
                    join_eps  = float(globals().get("JOIN_PRICE_ELASTICITY", -0.6))  # negative
                    churn_eps = float(globals().get("CHURN_PRICE_ELASTICITY",  0.3))  # positive

                    def _pmult(p, pref, eps):
                        if pref <= 0:
                            return 1.0
                        m = (max(p, 1e-9) / pref) ** eps
                        return float(np.clip(m, 0.25, 4.0))  # safety caps

                    price_mult_joins = _pmult(price, reference_price, join_eps)
                    price_mult_churn = _pmult(price, reference_price, churn_eps)
                    
                    for month in range(MONTHS):
                        
                        # Reset SE wage base every January
                        if (month % 12) == 0:
                            se_ss_wage_base_used_ytd = 0.0
                            
                        grant_received = 0
                        seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                        is_downturn = (rng.random() < DOWNTURN_PROB_PER_MONTH)
                        join_mult  = DOWNTURN_JOIN_MULT  if is_downturn else 1.0
                        churn_mult = DOWNTURN_CHURN_MULT if is_downturn else 1.0
                        
                        # ----- Beginner classes (optional) -----
                        revenue_classes_gross = 0.0
                        classes_cost = 0.0
                        class_students_this_month = 0
                        revenue_classes = 0
    
                        # Classes (semester-aware if enabled)
                        if CLASSES_ENABLED and _is_class_month(month):
                            # stochastic fill around mean
                            for _ in range(int(CLASS_COHORTS_PER_MONTH)):
                                fill = rng.normal(CLASS_FILL_MEAN, 0.08)
                                fill = float(np.clip(fill, 0.0, 1.0))
                                seats = int(round(CLASS_CAP_PER_COHORT * fill))
                                class_students_this_month += seats
                                revenue_classes_gross += seats * CLASS_PRICE
                                classes_cost += (seats * CLASS_COST_PER_STUDENT) + (CLASS_INSTR_RATE_PER_HR * CLASS_HOURS_PER_COHORT)
                
                            # schedule conversion of a fraction of students to members after a lag
                            if 'pending_class_conversions' not in locals():
                                pending_class_conversions = {}
                            target_m = month + int(CLASS_CONV_LAG_MO)
                            converts = int(round(class_students_this_month * CLASS_CONV_RATE))
                            if converts > 0:
                                pending_class_conversions[target_m] = pending_class_conversions.get(target_m, 0) + converts
                
                            # Net class revenue (adds into total_revenue)
                            revenue_classes = max(0.0, revenue_classes_gross - classes_cost)    
                        # conversions materialize this month    
                        class_joins_now = 0
                        if CLASSES_ENABLED and 'pending_class_conversions' in locals():
                            class_joins_now = int(pending_class_conversions.pop(month, 0))
                            # gate by available supply and MAX_ONBOARDINGS_PER_MONTH later
                        
                        # Replenish pools each month  <-- ADD THESE LINES
                        for _k, _v in MARKET_POOLS_INFLOW.items():
                            remaining_pool[_k] += int(_v)
    
                        # ----- Segment-based ramped adoption -----
                        cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                        capacity_damping = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
    
                        # Seasonality & downturn still matter
                        seasonal_mult = seasonal
                        downturn_join_mult = join_mult
                        awareness_mult = awareness_multiplier(month)
                        wom_mult = wom_multiplier(len(active_members))
    
                        # Unlock a new tranche of community-studio members at term boundaries
                        unlock_now = compute_cs_unlock_share(month, remaining_pool["community_studio"])
                        unlock_now = min(unlock_now, remaining_pool["community_studio"])
                        remaining_pool["community_studio"] -= unlock_now
                        cs_eligible += unlock_now
    
                        # Build per-pool effective monthly intent
                        # Add small lognormal noise to keep trajectories from looking too synthetic
                        noise = rng.lognormal(mean=-(ADOPTION_SIGMA**2)/2, sigma=ADOPTION_SIGMA)
                        
                        # --- price elasticity multiplier (cheaper than reference => higher intent)
                        price_mult = price_mult_joins
                        
                        intent_common_mult = (
                            seasonal_mult * downturn_join_mult * awareness_mult *
                            wom_mult * capacity_damping * noise * price_mult
                        )
                    
                        if JOIN_MODEL == "compartment":
                            lam_no   = max(0.0, BASELINE_RATE_NO_ACCESS * intent_common_mult)
                            lam_home = max(0.0, BASELINE_RATE_HOME      * intent_common_mult)
                            lam_comm = max(0.0, BASELINE_RATE_COMMUNITY * intent_common_mult)
                            pool_intents = {
                                "no_access":        _haz_to_prob(lam_no),
                                "home_studio":      _haz_to_prob(lam_home),
                                "community_studio": _haz_to_prob(lam_comm),  # applies only to cs_eligible
                            }
                        else:
                          # Tolerate bad overrides where POOL_BASE_INTENT is a scalar instead of a mapping
                          _base_int = POOL_BASE_INTENT
                          if not isinstance(_base_int, dict):
                              try:
                                  s = float(_base_int)
                                  _base_int = {
                                      "no_access": s,
                                      "home_studio": s,
                                      "community_studio": s,
                                  }
                              except Exception:
                                  # fall back to defaults if totally borked
                                  _base_int = {"no_access": 0.04, "home_studio": 0.01, "community_studio": 0.10}

                          pool_intents = {
                              "no_access":        _base_int["no_access"]        * intent_common_mult,
                              "home_studio":      _base_int["home_studio"]      * intent_common_mult,
                              "community_studio": _base_int["community_studio"] * intent_common_mult,  # applies only to cs_eligible
                          }
  
                        # Draw adopters from each pool
                        joins_no_access   = draw_adopters(remaining_pool["no_access"],      pool_intents["no_access"], rng)
                        joins_home        = draw_adopters(remaining_pool["home_studio"],    pool_intents["home_studio"], rng)
                        joins_comm_studio = draw_adopters(cs_eligible,                      pool_intents["community_studio"], rng)
    
                        # Update pools
                        remaining_pool["no_access"]   -= joins_no_access
                        remaining_pool["home_studio"] -= joins_home
                        cs_eligible                   -= joins_comm_studio
    
                        # Total joins this month (respect onboarding ops cap, if any)
                        joins = (
                             joins_no_access + joins_home + joins_comm_studio
                             + (int(stream.get("joins_from_workshops", np.zeros(MONTHS, dtype=int))[month]) if globals().get("WORKSHOPS_ENABLED", False) else 0)
                             + class_joins_now
                         )
                        
                        # --- referral loop (Poisson) ---
                        referral_joins = rng.poisson(REFERRAL_RATE_PER_MEMBER * len(active_members) * REFERRAL_CONV)
                        remaining_supply = remaining_pool["no_access"] + remaining_pool["home_studio"] + cs_eligible
                        referral_joins = int(min(referral_joins, remaining_supply))
    
                        bn_no_access = 0
                        bn_home = 0
                        bn_cs = 0
                        baseline_joins = 0

                        if JOIN_MODEL != "compartment":
                            # Baseline joins (capacity-aware trickle). Allocate like referrals, respecting remaining supply.
                            cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                            baseline_capacity_factor = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
                            baseline_demand = int(rng.poisson(BASELINE_JOIN_RATE * MEMBERSHIP_SOFT_CAP * baseline_capacity_factor))

                            remaining_supply = remaining_pool["no_access"] + remaining_pool["home_studio"] + cs_eligible
                            baseline_demand = min(baseline_demand, int(remaining_supply))

                            bn_no_access = min(baseline_demand, remaining_pool["no_access"])
                            remaining_pool["no_access"] -= bn_no_access
                            spill = baseline_demand - bn_no_access

                            bn_cs = min(spill, cs_eligible)
                            cs_eligible -= bn_cs
                            spill -= bn_cs

                            bn_home = min(spill, remaining_pool["home_studio"])
                            remaining_pool["home_studio"] -= bn_home

                            baseline_joins = bn_no_access + bn_cs + bn_home
                            joins += baseline_joins                    
                        
                        
                        
                        # apportion referrals
                        ref_no_access = min(referral_joins, remaining_pool["no_access"])
                        remaining_pool["no_access"] -= ref_no_access
                        spill = referral_joins - ref_no_access
    
                        ref_cs = min(spill, cs_eligible)
                        cs_eligible -= ref_cs
                        spill -= ref_cs
    
                        ref_home = min(spill, remaining_pool["home_studio"])
                        remaining_pool["home_studio"] -= ref_home
    
                        # final join count including class conversions
                        joins = (
                             class_joins_now
                             + (joins_no_access + joins_home + joins_comm_studio)
                             + baseline_joins
                             + referral_joins
                             + (int(stream.get("joins_from_workshops", np.zeros(MONTHS, dtype=int))[month]) if globals().get("WORKSHOPS_ENABLED", False) else 0)
                         )
                      
    
                     # If onboarding capped, roll back proportionally across ALL sources (incl baseline & referrals)
                        if MAX_ONBOARDINGS_PER_MONTH is not None and joins > MAX_ONBOARDINGS_PER_MONTH:
                            overflow = joins - MAX_ONBOARDINGS_PER_MONTH
                            
                            # Let classes soak overflow first (class converts don’t return to pools)
                            rb_classes = min(overflow, class_joins_now)
                            class_joins_now -= rb_classes
                            overflow -= rb_classes
    
                            # totals by source (organic + baseline + referral per pool)
                            take_no_access_total = (joins_no_access + bn_no_access + ref_no_access)
                            take_home_total      = (joins_home      + bn_home      + ref_home)
                            take_cs_total        = (joins_comm_studio + bn_cs      + ref_cs)
    
                            total_drawn = max(1, take_no_access_total + take_home_total + take_cs_total)
    
                            rb_no_access = int(round(overflow * (take_no_access_total / total_drawn)))
                            rb_home      = int(round(overflow * (take_home_total      / total_drawn)))
                            rb_cs        = overflow - rb_no_access - rb_home
    
                            # Return to pools
                            remaining_pool["no_access"]   += rb_no_access
                            remaining_pool["home_studio"] += rb_home
                            cs_eligible                   += rb_cs
    
                            # Reduce per-source takes (prefer rolling back referrals first, then baseline, then organic)
                            # --- no_access
                            give_from_ref = min(rb_no_access, ref_no_access); ref_no_access -= give_from_ref; rb_no_access -= give_from_ref
                            give_from_base = min(rb_no_access, bn_no_access); bn_no_access -= give_from_base; rb_no_access -= give_from_base
                            joins_no_access -= rb_no_access  # whatever remains comes from organic
    
                            # --- home
                            give_from_ref = min(rb_home, ref_home); ref_home -= give_from_ref; rb_home -= give_from_ref
                            give_from_base = min(rb_home, bn_home); bn_home -= give_from_base; rb_home -= give_from_base
                            joins_home -= rb_home
    
                            # --- community studio
                            give_from_ref = min(rb_cs, ref_cs); ref_cs -= give_from_ref; rb_cs -= give_from_ref
                            give_from_base = min(rb_cs, bn_cs); bn_cs -= give_from_base; rb_cs -= give_from_base
                            joins_comm_studio -= rb_cs
    
                            # Recompute aggregates post-rollback
                            baseline_joins = bn_no_access + bn_home + bn_cs
                            referral_joins = ref_no_access + ref_home + ref_cs
                            joins = (
                                class_joins_now
                                + (joins_no_access + joins_home + joins_comm_studio)
                                + baseline_joins
                                + referral_joins
                            )
                            # Safety
                            joins = min(joins, MAX_ONBOARDINGS_PER_MONTH)
    
                        # Create new members (keep archetype mix; NO geometric duration now)
                        # Cap joins so we never exceed MAX_MEMBERS
                        # Cap joins so we never exceed MAX_MEMBERS
                        joins = min(joins, MAX_MEMBERS - len(active_members))
                        
                        # Tag class converts for provenance (first N new members this month)
                        n_from_class = int(locals().get("class_joins_now", 0) or 0)
                        
                        for i in range(int(joins)):
                            archetype = rng.choice(
                                list(MEMBER_ARCHETYPES.keys()),
                                p=[v["prob"] for v in MEMBER_ARCHETYPES.values()]
                            )
                            active_members.append({
                                "type": archetype,
                                "start_month": month,
                                "monthly_fee": float(price),
                                "clay_bags": MEMBER_ARCHETYPES[archetype]["clay_bags"],
                                # NEW: record source to enable later analytics/retention tweaks
                                "src": "class" if i < n_from_class else "other",
                            })
    
                        # Tenure-based churn with utilization uplift near/over capacity (+ seasonality)
                        before = len(active_members)
                        kept = []
                        util_over = max(0.0, (len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)) - 1.0)
    
                        # seasonal multiplier for this calendar month (0-based month in the sim)
                        scm = seasonal_churn_mult(month)
    
                        for m in active_members:
                            tenure = month - m["start_month"]
    
                            p_leave = month_churn_prob(m["type"], tenure_mo=tenure)
                            p_leave *= churn_mult                         # downturn regime
                            p_leave *= price_mult_churn 
                            p_leave *= (1.0 + UTILIZATION_CHURN_UPLIFT * util_over)  # crowding
                            p_leave *= scm                                # 🔸 seasonality
                            p_leave = float(np.clip(p_leave, 0.0, 0.99))
    
                            if rng.random() > p_leave:
                                kept.append(m)
    
                        active_members = kept

                        # If UI provided a manual membership curve, enforce target headcount this month
                        manual_added = 0
                        manual_removed = 0
                        try:
                            _use_manual = bool(globals().get("USE_MANUAL_MEMBERSHIP_CURVE", False))
                            _manual_curve = globals().get("MANUAL_MEMBERSHIP_CURVE", None)
                        except Exception:
                            _use_manual, _manual_curve = False, None
                        if _use_manual and isinstance(_manual_curve, (list, tuple)) and month < len(_manual_curve):
                            target_n = int(max(0, float(_manual_curve[month])))
                            cur_n = len(active_members)
                            if target_n > cur_n:
                                # add delta members using archetype mix
                                delta = target_n - cur_n
                                labels = list(MEMBER_ARCHETYPES.keys())
                                probs = np.array([MEMBER_ARCHETYPES[k].get("prob", 0.0) for k in labels], dtype=float)
                                s = probs.sum()
                                probs = (probs / s) if s > 0 else np.full(len(labels), 1.0 / max(1, len(labels)))

                                for _ in range(delta):
                                    arch = rng.choice(labels, p=probs)
                                    active_members.append({
                                        "type": arch,
                                        "start_month": month,
                                        "monthly_fee": float(price),
                                        "clay_bags": MEMBER_ARCHETYPES[arch]["clay_bags"],
                                        "src": "manual"
                                    })
                                manual_added = delta
                            elif target_n < cur_n:
                                # randomly remove surplus to hit target
                                delta = cur_n - target_n
                                if delta > 0:
                                    drop_idx = set(rng.choice(np.arange(cur_n), size=delta, replace=False).tolist())
                                    active_members = [m for i, m in enumerate(active_members) if i not in drop_idx]
                                    manual_removed = delta                                    
                    
                        # Include churn plus any manual removals; count manual additions as joins for accounting
                        departures = (before - len(kept)) + manual_removed
                        net_adds = (joins + manual_added) - departures

                        # Revenues — membership, clay, firing, events
                        revenue_membership = sum(m["monthly_fee"] for m in active_members)
                        revenue_clay = 0.0  #gross; net margin after COGS below
                        revenue_firing = 0.0
                        total_clay_lbs = 0.0
                        
                       # Designated artist studios (stochastic monthly occupancy)
                        ds_occupied = int(rng.binomial(DESIGNATED_STUDIO_COUNT, DESIGNATED_STUDIO_BASE_OCCUPANCY)) if DESIGNATED_STUDIO_COUNT > 0 else 0
                        revenue_designated_studios = ds_occupied * DESIGNATED_STUDIO_PRICE
                     
                        for m in active_members:  
                            bags = rng.choice(m["clay_bags"])
                            revenue_clay += bags * RETAIL_CLAY_PRICE_PER_BAG
                            clay_lbs = bags * 25
                            total_clay_lbs += clay_lbs
                            revenue_firing += compute_firing_fee(clay_lbs)
    
                        # ----- Events: gross revenue and explicit COGS (mugs + consumables + optional labor) -----
                        revenue_events_gross = 0.0
                        events_cost_materials = 0.0
                        events_cost_labor = 0.0

                        events_this_month = 0
                        if events_enabled:
                            # seasonality: keep your existing normalization
                            seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                            # stochastic event count with hard cap
                            lam = max(0.0, base_lambda * seasonal)
                            events_this_month = int(np.clip(rng.poisson(lam), 0, events_max_per_month))

                            for _ in range(events_this_month):
                                attendees = int(rng.choice(attendees_range))
                                # revenue
                                event_gross = attendees * ticket_price
                                revenue_events_gross += event_gross
                                # materials (mugs + consumables)
                                mugs_cost = attendees * rng.uniform(*mug_cost_range)
                                consumables_cost = attendees * consumables_pp
                                events_cost_materials += (mugs_cost + consumables_cost)
                                # labor (optional)
                                if staff_rate_hr > 0 and hours_per_event > 0:
                                    events_cost_labor += staff_rate_hr * hours_per_event

                        revenue_events = max(0.0, revenue_events_gross - events_cost_materials - events_cost_labor)
                        
                        # Variable costs
                        variable_clay_cost = (total_clay_lbs / 25) * WHOLESALE_CLAY_COST_PER_BAG
                        variable_clay_cost *= float(globals().get("CLAY_COGS_MULT", 1.0))
                        water_cost = total_clay_lbs / 25 * GALLONS_PER_BAG_CLAY * WATER_COST_PER_GALLON
    
                        
                        # Electricity (De-Staged): turn on second-kiln draw only when at least 2 kilns purchased
                        kiln2_on = (_dyn_KILN_COUNT >= 2)
    
                        firings = firings_this_month(len(active_members))
                        kwh_per_firing = KWH_PER_FIRING_KMT1027 + (KWH_PER_FIRING_KMT1427 if kiln2_on else 0)
                        electricity_cost = firings * kwh_per_firing * COST_PER_KWH
    
                        # Heating
                        monthly_heating_cost = HEATING_COST_WINTER if month % 12 in [10, 11, 0, 1, 2, 3] else HEATING_COST_SUMMER
    
                        #Staff cost after expansion
                        staff_cost = STAFF_COST_PER_MONTH if len(active_members) >= STAFF_EXPANSION_THRESHOLD else 0.0
                                                 
                         #Maintenance
                        maintenance_cost = MAINTENANCE_BASE_COST + max(0, rng.normal(0, MAINTENANCE_RANDOM_STD))
                        maintenance_cost += float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0)) + float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))

                        #Marketing
                        if month < MARKETING_RAMP_MONTHS:
                            marketing_cost = MARKETING_COST_BASE * MARKETING_RAMP_MULTIPLIER
                        else:
                            marketing_cost = MARKETING_COST_BASE
                        
                       # ---------- S-corp owner salary (expense) & employer payroll taxes ----------
                        owner_salary_expense = 0.0
                        employer_payroll_tax = 0.0
                        if ENTITY_TYPE == "s_corp":
                            owner_salary_expense = SCORP_OWNER_SALARY_PER_MONTH
                            employer_payroll_tax = owner_salary_expense * EMPLOYER_PAYROLL_TAX_RATE
    
                        # Employee-side FICA withheld from wages (also remitted in cash by the business)
                        employee_withholding = 0.0
                        if ENTITY_TYPE == "s_corp":
                            employee_withholding = owner_salary_expense * EMPLOYEE_PAYROLL_TAX_RATE
    
                        # ---------- OpEx (pre-tax) ----------
                        # Annual rent increase (compounded once per year)
                        _rent_growth = float(globals().get("RENT_GROWTH_PCT", 0.0))/100
                        _year_index = (month // 12)
                        rent_this_month = fixed_rent * ((1.0 + _rent_growth) ** _year_index)
                        
                        # ---------- OpEx (pre-tax) ----------
                        fixed_opex_profit = rent_this_month + INSURANCE_COST + GLAZE_COST_PER_MONTH + monthly_heating_cost
                        total_opex_profit = (
                            fixed_opex_profit
                            + variable_clay_cost
                            + water_cost
                            + electricity_cost
                            + staff_cost
                            + marketing_cost
                            + maintenance_cost
                            + owner_salary_expense
                            + employer_payroll_tax
                        )
    
                        in_draw_window = in_owner_draw_window(month)  # existing calendar gate (start/end months)
                        within_stipend_quota = (month < OWNER_STIPEND_MONTHS)  # stipend only for first N months
                        owner_draw_now = owner_draw if (in_draw_window and within_stipend_quota) else 0.0
                        fixed_opex_cash = fixed_opex_profit + loan_payment_total_ts[month] + owner_draw_now

    
                        # Cash OpEx (pre-tax)
                        total_opex_cash = (
                            fixed_opex_cash
                            + variable_clay_cost
                            + water_cost
                            + electricity_cost
                            + staff_cost
                            + maintenance_cost
                            + marketing_cost
                            + owner_salary_expense
                            + employer_payroll_tax
                        )
                        
                            
                        total_revenue = (
                            revenue_membership + revenue_clay + revenue_firing + revenue_events
                            + (
                                float(stream.get("workshop_revenue", np.zeros(MONTHS))[month])
                                if globals().get("WORKSHOPS_ENABLED", False) else 0.0
                            )
                            + revenue_designated_studios
                            + (0.0 if not CLASSES_ENABLED else revenue_classes)
                        )
    
                        # ---------- Operating profit (pre-tax) ----------
                        op_profit = total_revenue - total_opex_profit
    
                        # ---------- Pass-through vs corporate tax accrual ----------
                        se_tax_this_month = 0.0
                        state_income_tax_this_month = 0.0
                        corp_tax_this_month = 0.0
    
                        if ENTITY_TYPE in ("sole_prop", "partnership"):
                            se_earnings = max(0.0, op_profit) * SE_EARNINGS_FACTOR
                            ss_base_remaining = max(0.0, SE_SOC_SEC_WAGE_BASE - se_ss_wage_base_used_ytd)
                            ss_taxable_now = min(se_earnings, ss_base_remaining)
                            se_tax_ss = ss_taxable_now * SE_SOC_SEC_RATE
                            se_ss_wage_base_used_ytd += ss_taxable_now
    
                            se_tax_medicare = se_earnings * SE_MEDICARE_RATE
                            se_tax_this_month = se_tax_ss + se_tax_medicare
                            se_tax_payable_accum += se_tax_this_month
    
                            half_se_deduction = 0.5 * se_tax_this_month
                            ma_taxable_income = max(0.0, op_profit - half_se_deduction)
                            state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                            state_tax_payable_accum += state_income_tax_this_month
    
                        elif ENTITY_TYPE == "s_corp":
                            # Owner salary + employer payroll tax already included in OpEx above.
                            ma_taxable_income = max(0.0, op_profit)
                            state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                            state_tax_payable_accum += state_income_tax_this_month
    
                        elif ENTITY_TYPE == "c_corp":
                            corp_taxable_income = max(0.0, op_profit)
                            corp_tax_this_month = corp_taxable_income * (FED_CORP_TAX_RATE + MA_CORP_TAX_RATE)
                            corp_tax_payable_accum += corp_tax_this_month
    
                        # ---------- Annual personal property tax (cash only unless you prefer accrual) ----------
                        property_tax_this_month = 0.0
                        if PERSONAL_PROPERTY_TAX_ANNUAL > 0 and ((month + 1) % 12 == (PERSONAL_PROPERTY_TAX_BILL_MONTH % 12)):
                            property_tax_this_month = PERSONAL_PROPERTY_TAX_ANNUAL/12
    
                        # ---------- Quarterly remittances (cash) ----------
                        tax_payments_this_month = 0.0
                        if ((month + 1) % ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                            if ENTITY_TYPE in ("sole_prop", "partnership", "s_corp"):
                                tax_payments_this_month += se_tax_payable_accum
                                tax_payments_this_month += state_tax_payable_accum
                                se_tax_payable_accum = 0.0
                                state_tax_payable_accum = 0.0
                            if ENTITY_TYPE == "c_corp":
                                tax_payments_this_month += corp_tax_payable_accum
                                corp_tax_payable_accum = 0.0
    
                        # Sales tax (retail clay here) — collected and remitted (cash only)
                        sales_tax_collected = revenue_clay * MA_SALES_TAX_RATE
                        sales_tax_payable_accum += sales_tax_collected
                        sales_tax_remitted = 0.0
                        if ((month + 1) % SALES_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                            sales_tax_remitted = sales_tax_payable_accum
                            tax_payments_this_month += sales_tax_remitted
                            sales_tax_payable_accum = 0.0
    
                        # ---------- Cash view ----------
                        total_opex_cash += property_tax_this_month
                        total_opex_cash += tax_payments_this_month - sales_tax_remitted
                        net_cash_flow = total_revenue - total_opex_cash + sales_tax_collected
    
                        # Accrual after-tax profit (income/self-employment/corp taxes only)
                        tax_cost = se_tax_this_month + state_income_tax_this_month + corp_tax_this_month
                        op_profit_after_tax = op_profit - tax_cost
    
                        # DSCR based on pre-tax operating profit (keep if you want to track it)
                        dscr = (op_profit / loan_payment_total_ts[month]) if loan_payment_total_ts[month] > 0 else np.nan
                        
                        # --- CFADS / Cash-DSCR (lender standard) ---
                        # total_opex_cash currently INCLUDES:
                        #   • cash operating costs (fixed + variable + payroll items)
                        #   • cash tax remittances (quarterly, etc.)
                        #   • owner_draw_now
                        #   • monthly_loan_payment
                        # We want CFADS to EXCLUDE owner draws and EXCLUDE debt service,
                        # but INCLUDE cash taxes and all other cash opex.
                        
                        opex_cash_excl_debt_and_draws = (
                            total_opex_cash
                            - loan_payment_total_ts[month]    # remove debt service
                            - owner_draw_now                  # remove distributions
                        )
                        
                        cfads = total_revenue - opex_cash_excl_debt_and_draws
                        
                        dscr_cash = (cfads / loan_payment_total_ts[month]) if loan_payment_total_ts[month] > 0 else np.nan
    
                        # Month 0 loan proceeds (de-staged)
                        if month == 0:
                            # Month 0: add only the upfront portions for each loan
                            if month == 0:
                                if CAPEX_MODE == "upfront" and loan_504_principal > 0:
                                    cash_balance += float(loan_504_principal)
                                if OPEX_MODE == "upfront" and loan_7a_principal > 0:
                                    cash_balance += float(loan_7a_principal)
           
                                # No "upfront CapEx" subtraction — CapEx spends only when CAPEX_ITEMS fire
                                if 'fees_cash_outflow' in locals() and fees_cash_outflow > 0:
                                    cash_balance -= float(fees_cash_outflow)
                            else:
                                # staged: no proceeds at t=0; draws occur when purchases execute
                                pass
                            
                        # Generalized staged CapEx (month-based or membership-based)
                        capex_draw_this_month = 0.0
                        loan_tranche_draw_capex  = 0.0
                        loan_tranche_draw_opex  = 0.0
                        
                        if _capex_queue:
                            current_members = len(active_members)
                            for _item in _capex_queue:
                                if _item["purchased"]:
                                    continue
                                m_ok = (_item["month"] is not None) and (month == int(_item["month"]))
                                n_ok = (_item["member_threshold"] is not None) and (current_members >= int(_item["member_threshold"]))
                                if m_ok or n_ok:
                                    cnt = int(_item.get("count", 1) or 1)
                                    unit = float(_item.get("unit_cost", 0) or 0.0)
                                    total_cost = unit * cnt
                                    capex_draw_this_month += total_cost
                                    
                                    # ---- Apply equipment effects based on label ----
                                    lbl = str(_item.get("label", "")).lower()
                                    cnt = int(_item.get("count", 1) or 1)
                                    # Kilns: track count (drives electricity for kiln #2)
                                    if "kiln" in lbl:
                                        _dyn_KILN_COUNT += max(1, cnt)
                                    # Wheels: add capacity by count
                                    if "wheel" in lbl:
                                        if "wheels" in _dyn_STATIONS:
                                            curr = int(_dyn_STATIONS["wheels"].get("capacity", 0))
                                            target = max(0, cnt)
                                            _dyn_STATIONS["wheels"]["capacity"] = max(curr, target)

                                    # Wire racks: interpret count as TOTAL racks target (not incremental)
                                    if "rack" in lbl:
                                        target_members = 3 * max(0, cnt)   # 3 members per rack
                                        _dyn_MAX_MEMBERS = max(_dyn_MAX_MEMBERS, target_members)

                                    # Slab roller: boost handbuilding capacity (+20%) and add maintenance
                                    if "slab" in lbl and "roll" in lbl:
                                        if "handbuilding" in _dyn_STATIONS:
                                            hb = int(_dyn_STATIONS["handbuilding"].get("capacity", 6))
                                            _dyn_STATIONS["handbuilding"]["capacity"] = max(1, int(round(hb * 1.20)))
                                        _dyn_SLAB_MAINT += 10.0
                                    # Pug mill: reduce clay COGS multiplier and add maintenance
                                    if "pug" in lbl:
                                        _dyn_CLAY_COGS_MULT = 0.75
                                        _dyn_PUGMILL_MAINT += 20.0
                                    _item["purchased"] = True
                            if capex_draw_this_month > 0.0:
                                cash_balance -= capex_draw_this_month
                                cumulative_after_capex -= capex_draw_this_month
                                # If staged loans are enabled, draw a tranche and install its schedule starting next month
                            # Compute staged CapEx tranche amount from this month's purchase
                            tranche = float(_capex_draws_ts[month]) if (0 <= month < len(_capex_draws_ts)) else 0.0

                            if CAPEX_MODE == "staged":
                                # CapEx staged tranche
                                cash_balance += tranche
                                loan_tranche_draw_capex = tranche
                                _add_staged_tranche_into_array(
                                    loan_payment_504_ts,
                                    month,
                                    tranche,
                                    LOAN_504_ANNUAL_RATE,
                                    LOAN_504_TERM_YEARS,
                                    IO_MONTHS_504,
                                    MONTHS,
                                )
                                loan_payment_total_ts = loan_payment_504_ts + loan_payment_7a_ts
                                
                            if OPEX_MODE == "staged":
                                rule = globals().get("LOAN_STAGED_RULE_OPEX", {}) or {}
                                facility_limit = float(rule.get("facility_limit", 0.0) or 0.0)
                                min_draw = float(rule.get("min_draw", 0.0) or 0.0)
                                max_draw_raw = rule.get("max_draw", None)
                                max_draw = None if (max_draw_raw in (None, 0)) else float(max_draw_raw)
                                reserve_floor = float(rule.get("reserve_floor", 0.0) or 0.0)
                                if cash_balance < reserve_floor and opex_remaining > 0:
                                    needed = reserve_floor - cash_balance
                                    draw_amt = max(needed, min_draw)
                                    if max_draw is not None:
                                        draw_amt = min(draw_amt, max_draw)
                                    draw_amt = min(draw_amt, opex_remaining)
                                    if draw_amt > 0:
                                        cash_balance += draw_amt
                                        opex_remaining -= draw_amt
                                        loan_tranche_draw_opex = draw_amt
                                        _add_staged_tranche_into_array(
                                            loan_payment_7a_ts,
                                            month,
                                            draw_amt,
                                            LOAN_7A_ANNUAL_RATE,
                                            LOAN_7A_TERM_YEARS,
                                            IO_MONTHS_7A,
                                            MONTHS,
                                        )
                                        loan_payment_total_ts = loan_payment_504_ts + loan_payment_7a_ts
                            
                            # ---- Write back dynamic globals so downstream code sees the changes ----
                            STATIONS.update(_dyn_STATIONS)
                            globals()["MAX_MEMBERS"] = int(_dyn_MAX_MEMBERS)
                            globals()["CLAY_COGS_MULT"] = float(_dyn_CLAY_COGS_MULT)
                            globals()["PUGMILL_MAINT_COST_PER_MONTH"] = float(_dyn_PUGMILL_MAINT)
                            globals()["SLAB_ROLLER_MAINT_COST_PER_MONTH"] = float(_dyn_SLAB_MAINT)
    
                        # Apply monthly results
                        cash_balance += net_cash_flow
                        cumulative_op_profit += op_profit
                        cumulative_after_capex += op_profit
    
                        # Insolvency before grant
                        # FIX
                        pre_grant_or_no_grant = (grant_month is None) or (month < grant_month)
                        if pre_grant_or_no_grant and (cash_balance < 0) and (not insolvent_before_grant):
                            insolvent_before_grant = True
    
                        # Grant (keep 0-based)
                        if (grant_month is not None) and (month == grant_month):
                            cash_balance += grant_amount
                            grant_received = grant_amount
                        
                        # --- derive workshop stats for this month (from globals) ---
                        _wpm   = float(globals().get("WORKSHOPS_PER_MONTH", 0.0))
                        _avg_n = int(globals().get("WORKSHOP_AVG_ATTENDANCE", 0))
                        _fee   = float(globals().get("WORKSHOP_FEE", 0.0))
                        _cost  = float(globals().get("WORKSHOP_COST_PER_EVENT", 0.0))

                        # If workshops are globally disabled, force zeros
                        if not bool(globals().get("WORKSHOPS_ENABLED", False)):
                            _wpm = _avg_n = _fee = _cost = 0.0

                        _workshop_attendees = int(round(_wpm * _avg_n))
                        _gross_ws = float(_workshop_attendees * _fee)
                        _cost_ws  = float(_wpm * _cost)
                        
                         # --- Update running loan balances (after any staged draws this month) ---
                        _draw504 = float(loan_tranche_draw_capex) if 'loan_tranche_draw_capex' in locals() else 0.0
                        _draw7a  = float(loan_tranche_draw_opex)  if 'loan_tranche_draw_opex'  in locals() else 0.0
                        _pay504  = float(loan_payment_504_ts[month]) if month < len(loan_payment_504_ts) else 0.0
                        _pay7a   = float(loan_payment_7a_ts[month])  if month < len(loan_payment_7a_ts)  else 0.0
                        # Add any draws first
                        _bal504 += _draw504
                        _bal7a  += _draw7a
                         # Split payment into interest + principal on current balances
                        _int504  = _bal504 * _r504
                        _int7a   = _bal7a  * _r7a
                        _prin504 = max(0.0, _pay504 - _int504)
                        _prin7a  = max(0.0, _pay7a  - _int7a)
                        _bal504  = max(0.0, _bal504 - _prin504)
                        _bal7a   = max(0.0, _bal7a  - _prin7a)
                        loan_balance_504_ts[month] = _bal504
                        loan_balance_7a_ts[month]  = _bal7a
                        
                        # Store row
                        rows.append({
                            "simulation_id": sim,
                            "scenario": scen_name,
                            "rent": fixed_rent,
                            "owner_draw": owner_draw,
                            "month": month + 1,
                            "active_members": len(active_members),
                            "joins": joins,
                            "departures": departures,
                            "net_adds": net_adds,
                            "cash_balance": cash_balance,
                            "net_cash_flow": net_cash_flow,
                            "cumulative_op_profit": cumulative_op_profit,
                            "cumulative_profit_after_capex": cumulative_after_capex,
                            "revenue_membership": revenue_membership,
                            "revenue_firing": revenue_firing,
                            "revenue_clay": revenue_clay,
                            "revenue_events": revenue_events,
                            "revenue_workshops_net": float(stream["workshop_revenue"][month]),
                            "revenue_designated_studios": revenue_designated_studios,
                            "designated_studio_occupied": ds_occupied,
                            "grant_received": grant_received,
                            "insolvent_before_grant": insolvent_before_grant,
                            "grant_month": grant_month,
                            "grant_amount": grant_amount,
                            "is_downturn": is_downturn,                        
                            "loan_payment_total": float(loan_payment_total_ts[month]),
                            "loan_payment_504": float(loan_payment_504_ts[month]),
                            "loan_payment_7a": float(loan_payment_7a_ts[month]),
                            "loan_principal_total": loan_principal_total,
                            "loan_principal_504": loan_504_principal,
                            "loan_principal_7a": loan_7a_principal,
                            # Fees visibility (only meaningful at month 0; still included for traceability)
                            "fees_cash_outflow": float(fees_cash_outflow if 'fees_cash_outflow' in locals() else 0.0),
                            "fees_7a_financed": float((fees_7a_total if ('fees_7a_total' in locals() and FINANCE_FEES_7A) else 0.0)),
                            "fees_504_financed": float((fees_504_total if ('fees_504_total' in locals() and FINANCE_FEES_504) else 0.0)),  
                            "capex_I_cost": capex_I_cost,
                            "capex_II_cost": capex_II_cost,
                            "capex_draw": float(capex_draw_this_month) if 'capex_draw_this_month' in locals() else 0.0,
                            "loan_tranche_draw_capex": float(loan_tranche_draw_capex) if 'loan_tranche_draw_capex' in locals() else 0.0,
                            "loan_tranche_draw_opex":  float(loan_tranche_draw_opex)  if 'loan_tranche_draw_opex'  in locals() else 0.0,
                            "runway_costs": sized_runway_costs,
                            "loan_balance_504": float(loan_balance_504_ts[month]),
                            "loan_balance_7a": float(loan_balance_7a_ts[month]),
                            "dscr": dscr,
                            "dscr_cash": dscr_cash,
                            "staff_cost": staff_cost,
                            "maintenance_cost": maintenance_cost,
                            "marketing_cost": marketing_cost,
                            "op_profit_after_tax": op_profit_after_tax,
                            "sales_tax_collected": sales_tax_collected,
                            "sales_tax_remitted": sales_tax_remitted,
                            "se_tax_accrued": se_tax_this_month,
                            "state_income_tax_accrued": state_income_tax_this_month,
                            "corp_tax_accrued": corp_tax_this_month,
                            "tax_payments_made": tax_payments_this_month,
                            "property_tax": property_tax_this_month,
                            "owner_salary_expense": owner_salary_expense,
                            # staged-only: amount of loan tranche drawn this month (0 in upfront months)
                            "employer_payroll_tax": employer_payroll_tax,
                            "entity_type": ENTITY_TYPE,
                            "owner_draw_paid": owner_draw_now,
                            "employee_withholding": employee_withholding,
                            "workshop_attendees": _workshop_attendees,
                            "workshop_gross": _gross_ws,
                            "workshop_cost": _cost_ws,
                            "events_this_month": events_this_month,
                            "revenue_events_gross": revenue_events_gross,
                            "events_cost_materials": events_cost_materials,
                            "events_cost_labor": events_cost_labor,
                            "revenue_classes_gross": revenue_classes_gross,
                            "classes_cost": classes_cost,
                            "revenue_classes": revenue_classes,
                            "class_students": class_students_this_month,
                            "cfads":cfads,
                            "dscr_cash_breach_1_00": (dscr_cash < 1.00) if np.isfinite(dscr_cash) else False,
                            "dscr_cash_breach_1_25": (dscr_cash < DSCR_CASH_TARGET) if np.isfinite(dscr_cash) else False,
                        })
    
    # ---- Build DataFrame ----
    results_df = pd.DataFrame(rows)
    print("Built results_df with shape:", results_df.shape)
    
    # =============================================================================
    # Dashboard Plots
    # =============================================================================
    sns.set_context("talk")
    
    # Global membership (median + band) with cap
    g = results_df.groupby("month")["active_members"]
    med = g.median(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)
     
    # Cash balance overlays per (scenario, rent)
    for scen in results_df["scenario"].unique():
        for rent_val in sorted(results_df["rent"].unique()):
            df_rent = results_df[(results_df["scenario"] == scen) & (results_df["rent"] == rent_val)]
            if df_rent.empty:
                continue
    
            fig, ax = plt.subplots(figsize=(10, 6))
            for od, df_od in df_rent.groupby("owner_draw"):
                grouped = df_od.groupby("month")["cash_balance"]
                median = grouped.median()
                p10 = grouped.quantile(0.1)
                p90 = grouped.quantile(0.9)
    
                ax.plot(median.index, median.values, label=f"Draw ${od:,.0f}/mo", linewidth=2)
                ax.fill_between(median.index, p10.values, p90.values, alpha=0.10)
    
            # Grant markers
            grant_info = df_rent[["grant_month", "grant_amount"]].drop_duplicates()
            for _, row in grant_info.iterrows():
                gm, ga = row["grant_month"], row["grant_amount"]
                if pd.notna(gm) and gm is not None:
                    ax.axvline(gm+1, color="green", linestyle="--", linewidth=1.5)
                    ymin, ymax = ax.get_ylim()
                    ax.text(gm+1, ymin + 0.05*(ymax - ymin), f"Grant ${ga:,.0f}", color="green", rotation=90, va="bottom")
    
            ax.axhline(0, color="black", linestyle=":", linewidth=1)
            ax.set_title(f"Cash Balance Over Time — {scen} | Rent ${rent_val:,.0f}/mo")
            ax.set_xlabel("Month"); ax.set_ylabel("Cash Balance ($)")
            ax.legend(loc="best"); plt.tight_layout(); plt.show()
    
    # Operating break-even heatmaps
    def first_break_even(group):
        be_month = group.loc[group["cumulative_op_profit"] >= 0, "month"].min()
        return be_month if pd.notna(be_month) else np.nan
    
    be_df = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .apply(first_break_even)
        .reset_index(name="op_break_even_month")
    )
    
    for scen in be_df["scenario"].unique():
        pivot = be_df[be_df["scenario"] == scen].pivot_table(
            index="owner_draw", columns="rent", values="op_break_even_month", aggfunc="median"
        )
        # Skip empty / all-NaN tables to avoid seaborn crash
        if pivot.size == 0 or pivot.dropna(how="all").empty:
            continue
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Months"})
        
        
        plt.title(f"Median Months to Operating Break-Even — {scen}")
        plt.xlabel("Monthly Rent ($)"); plt.ylabel("Owner Draw ($/mo)")
        plt.tight_layout(); plt.show()
    
    # Revenue vs OpEx (cash) small-multiples, including workshops
    for scen in results_df["scenario"].unique():
        for rent_val in sorted(results_df["rent"].unique()):
            df_rent = results_df[(results_df["scenario"] == scen) & (results_df["rent"] == rent_val)]
            if df_rent.empty:
                continue
    
            owner_draws = sorted(df_rent["owner_draw"].unique())
            ncols = len(owner_draws); nrows = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.6*nrows), sharey=True, sharex=True)
            axes = np.atleast_2d(axes)
    
            for j, od in enumerate(owner_draws):
                ax = axes[0, j]
                df_sub = df_rent[df_rent["owner_draw"] == od].copy()
    
                rev_components = [
                    "revenue_membership",
                    "revenue_firing",
                    "revenue_clay",
                    "revenue_events",
                    "revenue_workshops_net",
                    "revenue_designated_studios",
                    "revenue_classes",  # NEW: classes (net)
                ]
                df_sub["total_revenue"] = df_sub[rev_components].sum(axis=1)
                df_sub["total_opex_cash"] = df_sub["total_revenue"] - df_sub["net_cash_flow"]

                g_rev = df_sub.groupby("month")[rev_components].median()
                g_ops = df_sub.groupby("month")[["total_revenue", "total_opex_cash", "net_cash_flow"]].median()

                ax.stackplot(
                    g_rev.index,
                    g_rev["revenue_membership"],
                    g_rev["revenue_firing"],
                    g_rev["revenue_clay"],
                    g_rev["revenue_events"],
                    g_rev["revenue_workshops_net"],
                    g_rev["revenue_designated_studios"],
                    g_rev["revenue_classes"],  # NEW
                    labels=[
                        "Membership", "Firing Fees", "Clay", "Events",
                        "Workshops (net)", "Designated Studios", "Classes (net)"  # NEW
                    ],
                    alpha=0.9,
                )
                
                ax.plot(g_ops.index, g_ops["total_opex_cash"], linewidth=2.0, label="Total OpEx (cash)")
                ax.plot(g_ops.index, g_ops["net_cash_flow"], linestyle="--", linewidth=1.5, label="Net Cash Flow")
    
                ax.set_title(f"Draw ${od:,.0f}/mo", fontsize=11)
                if j == 0: ax.set_ylabel("Dollars ($)")
                ax.set_xlabel("Month")
    
           # Collect and de‑duplicate legend entries from all facets
            for ax in fig.axes:
                # keep existing text but shrink font
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10)
                ax.tick_params(axis="both", which="major", labelsize=9)
                ax.tick_params(axis="both", which="minor", labelsize=8)  # if you have minor ticks
                
            handles_all, labels_all = [], []
            for ax_ in axes.flat:
                h_, l_ = ax_.get_legend_handles_labels()
                handles_all.extend(h_); labels_all.extend(l_)
            from collections import OrderedDict
            by_label = OrderedDict()
            for h_, l_ in zip(handles_all, labels_all):
                if l_ not in by_label:
                    by_label[l_] = h_
            # ↓ Make axis labels and tick labels smaller on all axes in this figure
            
            fig.legend(
                list(by_label.values()),
                list(by_label.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),   # push below x-axis
                ncol=min(4, len(by_label)),    # spread entries
                frameon=False,
                fontsize = 10
            )
            
            # Title + spacing that leaves headroom (top) and legend space (bottom)
            fig.suptitle(f"Revenue vs OpEx — {scen} | Rent ${rent_val:,.0f}/mo", y=0.98, fontsize=14)
            fig.tight_layout(rect=[0, 0.08, 1, 0.95])  # leave more room at bottom
            plt.show()
    
    
    # --- Spaghetti + band for one configuration ---
    scenario_pick   = results_df["scenario"].unique()[0]
    rent_pick       = sorted(results_df["rent"].unique())[0]
    owner_draw_pick = sorted(results_df["owner_draw"].unique())[0]
    
    cfg = results_df[
        (results_df["scenario"] == scenario_pick) &
        (results_df["rent"] == rent_pick) &
        (results_df["owner_draw"] == owner_draw_pick)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # median band
    g = cfg.groupby("month")["active_members"]
    med = g.median(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)
    ax.plot(med.index, med.values, linewidth=2, label="Median")
    ax.fill_between(med.index, p10.values, p90.values, alpha=0.12, label="10–90%")
    ax.axhline(MEMBERSHIP_SOFT_CAP, linestyle="--", linewidth=1.5, label=f"Soft cap ≈ {MEMBERSHIP_SOFT_CAP:.0f}")
    try:
        ax.axhline(MAX_MEMBERS, linestyle=":", linewidth=1.5, color="orange", label=f"Hard cap = {int(MAX_MEMBERS)}")
    except Exception:
        pass
    
    # random paths
    _from = cfg["simulation_id"].unique()
    local_rng = default_rng(SeedSequence([RANDOM_SEED, 999, int(rent_pick), int(owner_draw_pick)]))
    for sim_id in local_rng.choice(_from, size=min(25, len(_from)), replace=False):
        path = cfg[cfg["simulation_id"] == sim_id].sort_values("month")
        ax.plot(path["month"], path["active_members"], linewidth=0.8, alpha=0.35)
    
    ax.set_title(f"Membership Paths — {scenario_pick} | Rent ${rent_pick:,.0f}/mo | Draw ${owner_draw_pick:,.0f}/mo")
    ax.set_xlabel("Month"); ax.set_ylabel("Active Members")
    ax.legend(loc="best"); plt.tight_layout(); plt.show()
            
            
    # Net adds histograms + probability of negative months (global simplified view)
    vals = results_df["net_adds"].values
    # plt.figure(figsize=(10,6))
    # plt.hist(vals, bins=range(int(vals.min())-1, int(vals.max())+2), alpha=0.5)
    # plt.axvline(0, color="black", linestyle=":")
    # plt.title("Distribution of Monthly Net Adds — All configurations")
    # plt.xlabel("Net Adds (Joins − Departures) per Month"); plt.ylabel("Frequency")
    # plt.tight_layout(); plt.show()
    
    neg_prob = (results_df["net_adds"] < 0).mean()
    print(f"Probability of a negative net-adds month (global): {neg_prob:.1%}")
    
    # Optional: Workshop timing diagnostic — fraction of months with workshops (by scenario)
    diag = (results_df.assign(ws=(results_df["revenue_workshops_net"]>0))
            .groupby(["scenario","simulation_id"])["ws"].mean().groupby("scenario").describe())
    print("\nWorkshop timing diagnostic (share of months with a workshop):")
    print(diag)
    
    # =============================================================================
    # Summary Table
    # =============================================================================
    insolvent_summary = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .agg({"insolvent_before_grant":"max"})
        .groupby(level=[0,1,2]).mean()
        .rename(columns={"insolvent_before_grant": "pct_insolvent_before_grant"})
    )
    
    be_df = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .apply(first_break_even)
        .reset_index(name="op_break_even_month")
    )
    
    beop_summary = (
        be_df
        .groupby(["scenario", "rent", "owner_draw"])["op_break_even_month"]
        .median()
        .to_frame("median_op_break_even_month")
    )
    
    final_cash_summary = (
        results_df[results_df["month"] == MONTHS]
        .groupby(["scenario", "rent", "owner_draw"])["cash_balance"]
        .median()
        .to_frame("median_final_cash_m60")
    )
    
    # Median minimum cash across the horizon (stress indicator)
    min_cash_summary = (
        results_df.groupby(["scenario","rent","owner_draw","simulation_id"])["cash_balance"].min()
        .groupby(level=[0,1,2]).median()
        .to_frame("median_min_cash")
    )
    
    # Median CFADS months 12 & 24
    cfads_12 = (results_df[results_df["month"]==12]
                .groupby(["scenario","rent","owner_draw"])["cfads"].median()
                .rename("median_cfads_m12"))
    cfads_24 = (results_df[results_df["month"]==24]
                .groupby(["scenario","rent","owner_draw"])["cfads"].median()
                .rename("median_cfads_m24"))
    
    # % months breaching cash-DSCR<1.25
    breach_rate = (results_df.groupby(["scenario","rent","owner_draw"])["dscr_cash_breach_1_25"]
                   .mean().rename("%_months_below_1_25"))
    
    
    summary_table = (insolvent_summary
                     .join(beop_summary, how="outer")
                     .join(final_cash_summary, how="outer")
                     .join(min_cash_summary, how="outer")
                     .join(cfads_12, on=["scenario","rent","owner_draw"])
                     .join(cfads_24, on=["scenario","rent","owner_draw"])
                     .join(breach_rate, on=["scenario","rent","owner_draw"]))
    
    
    
    # Add median monthly revenue from designated studios
    ds_rev_summary = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])["revenue_designated_studios"]
        .median()
        .groupby(level=[0, 1, 2]).median()
        .to_frame("median_monthly_ds_revenue")
    )
    summary_table = summary_table.join(
        ds_rev_summary,
        on=["scenario", "rent", "owner_draw"]
    )
    
    summary_table = summary_table.reset_index()
    print("\n=== Summary (by scenario, rent, owner_draw) ===")
    print(summary_table.to_string(index=False))
    
    # Optional: export CSVs for lender materials
    try:
        results_df.to_csv("gcws_results_detailed.csv", index=False)
        summary_table.to_csv("gcws_summary_table.csv", index=False)
        print("\nSaved: gcws_results_detailed.csv and gcws_summary_table.csv")
    except Exception as e:
        print("CSV export skipped:", e)
    
        
        
    # =============================================================================
    # Owner take-home summary (salary + draws − personal taxes − employee FICA)
    # =============================================================================
    def summarize_owner_takehome(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
    
        # employee FICA withheld from S‑corp wages (owner’s side)
        out["employee_withholding"] = np.where(
            out["entity_type"] == "s_corp",
            out["owner_salary_expense"] * EMPLOYEE_PAYROLL_TAX_RATE,
            0.0
        )
    
        # net wage paid to owner after employee FICA (only relevant for s_corp)
        out["wage_net_to_owner"] = out["owner_salary_expense"] - out["employee_withholding"]
    
        # pass‑through personal taxes we already accrued in the model
        # (sole prop / partnership: SE tax + MA personal; s‑corp: MA personal on pass-through)
        out["personal_taxes_owner"] = np.where(
            out["entity_type"].isin(["sole_prop", "partnership", "s_corp"]),
            out["se_tax_accrued"] + out["state_income_tax_accrued"],
            0.0
        )
    
        # make sure the column exists even if you forgot to add it
        if "owner_draw_paid" not in out.columns:
            out["owner_draw_paid"] = 0.0  # fallback
    
        # monthly owner take-home cash
        out["owner_takehome_cash"] = (
            out["wage_net_to_owner"] + out["owner_draw_paid"] - out["personal_taxes_owner"]
        )
    
        # aggregate to annual per simulation (12 months) and then take scenario medians
        annual = (
            out.groupby(["scenario", "rent", "owner_draw", "simulation_id"], as_index=False)
               .agg(owner_takehome_annual=("owner_takehome_cash", "sum"))
        )
        med = (annual
               .groupby(["scenario", "rent", "owner_draw"], as_index=False)["owner_takehome_annual"]
               .median()
               .rename(columns={"owner_takehome_annual": "median_owner_takehome_annual"}))
    
        return med
    
    owner_takehome_table = summarize_owner_takehome(results_df)
    print("\n=== Median Owner Take‑Home (annual, by scenario/rent/draw) ===")
    print(owner_takehome_table.to_string(index=False))
    
    # =============================================================================
    # Optional: Sweep EXTRA_BUFFER to keep median min-cash >= 0 (ask sizing helper)
    # =============================================================================
    def sweep_extra_buffer(buffers=(0, 5000, 10000, 15000, 20000)):
        """
        Coarse sensitivity on EXTRA_BUFFER added to the loan principal.
        Returns a DataFrame with median min-cash by (scenario, rent, owner_draw) for each buffer.
        This does not alter baseline results unless you manually set EXTRA_BUFFER and re-run.
        """
        import pandas as pd
        global EXTRA_BUFFER
        baseline_extra = EXTRA_BUFFER
        out = []
    
        for b in buffers:
            EXTRA_BUFFER = float(b)
    
            # --- Minimal re-run to capture min cash per simulation ---
    
            rows_min = []
            MEMBERSHIP_SOFT_CAP, _ = compute_membership_soft_cap()
    
            for fixed_rent in RENT_SCENARIOS:
                for owner_draw in OWNER_DRAW_SCENARIOS:
                    for scen_cfg in SCENARIO_CONFIGS:
                        scen_name = scen_cfg["name"]
                                                # --- Snapshot baseline mutable globals so each simulation starts identical ---
                        _BASE_STATIONS    = copy.deepcopy(STATIONS)
                        _BASE_MAX_MEMBERS = int(MAX_MEMBERS)
                        _BASE_CLAY_COGS   = float(globals().get("CLAY_COGS_MULT", 1.0))
                        _BASE_PUG_MAINT   = float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0))
                        _BASE_SLAB_MAINT  = float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))
                        
                        for sim in range(N_SIMULATIONS):
                            scen_index = next(i for i, s in enumerate(SCENARIO_CONFIGS) if s["name"] == scen_name)
                            ss = SeedSequence([RANDOM_SEED, int(fixed_rent), int(owner_draw), int(scen_index), int(sim)])
                            rng = default_rng(ss)
                            
                                                    # --- Reset mutable globals to the baseline for reproducibility ---
                            if isinstance(STATIONS, dict):
                                STATIONS.clear()
                                STATIONS.update(copy.deepcopy(_BASE_STATIONS))
                            else:
                                globals()["STATIONS"] = copy.deepcopy(_BASE_STATIONS)
                            globals()["MAX_MEMBERS"] = _BASE_MAX_MEMBERS
                            globals()["CLAY_COGS_MULT"] = _BASE_CLAY_COGS
                            globals()["PUGMILL_MAINT_COST_PER_MONTH"] = _BASE_PUG_MAINT
                            globals()["SLAB_ROLLER_MAINT_COST_PER_MONTH"] = _BASE_SLAB_MAINT
                            
                            # --- Pricing elasticity setup (fixed baseline) ---
                            price = float(globals().get("PRICE", 165.0))
                            reference_price = float(globals().get("REFERENCE_PRICE", price))  # default to current price if missing
                            join_eps  = float(globals().get("JOIN_PRICE_ELASTICITY", -0.6))  # negative
                            churn_eps = float(globals().get("CHURN_PRICE_ELASTICITY",  0.3))  # positive
                            def _pmult(p, pref, eps):
                                if pref <= 0: 
                                    return 1.0
                                m = (max(p, 1e-9) / pref) ** eps
                                return float(np.clip(m, 0.25, 4.0))  # safety caps
                            price_mult_joins = _pmult(price, reference_price, join_eps)
                            price_mult_churn = _pmult(price, reference_price, churn_eps)
                            # --- CapEx and loan sizing (aligned with main sim) ---
                            capex_I_cost = 0.0
                            capex_II_cost = 0.0
    
                            # Runway (INCLUDES owner draw, matching main sim)
                            avg_monthly_heat = (HEATING_COST_WINTER + HEATING_COST_SUMMER) / 2
                            runway_costs = (
                                INSURANCE_COST + GLAZE_COST_PER_MONTH + avg_monthly_heat + fixed_rent + owner_draw
                            ) * RUNWAY_MONTHS
    
                            capex_table_total = 0.0
                            for _it in CAPEX_ITEMS:
                                unit = float(_it.get("unit_cost", 0.0) or 0.0)
                                cnt  = int(_it.get("count", 1) or 1)
                                if unit > 0:
                                    capex_table_total += unit * max(1, cnt)
                                else:
                                    amt = _it.get("amount", None)
                                    if amt is not None:
                                        capex_table_total += float(amt)
    
                            # Split loans (504 = CapEx + contingency; 7(a) = runway + EXTRA_BUFFER)
                            total_capex_for_loan = capex_table_total


                            loan_504_principal = total_capex_for_loan * (1 + LOAN_CONTINGENCY_PCT)
                            loan_7a_principal  = runway_costs + EXTRA_BUFFER
        
                            # ---- SBA fees (can be financed into principal or paid in cash) ----
                            fees_7a_pct  = float(FEES_UPFRONT_PCT_7A)  * float(loan_7a_principal)
                            fees_504_pct = float(FEES_UPFRONT_PCT_504) * float(loan_504_principal)
                            flat_fees    = float(FEES_PACKAGING) + float(FEES_CLOSING)  # attach to 7(a) for simplicity
                            fees_7a_total  = fees_7a_pct  + flat_fees
                            fees_504_total = fees_504_pct
        
                            fees_cash_outflow = 0.0
                            if FINANCE_FEES_7A:
                                loan_7a_principal += fees_7a_total
                            else:
                                fees_cash_outflow += fees_7a_total
                            if FINANCE_FEES_504:
                                loan_504_principal += fees_504_total
                            else:
                                fees_cash_outflow += fees_504_total
                            # ---- end SBA fees ----
    
                            # Monthly debt service (define these; they’re used later)
                           # Build per-month payment schedules (IO -> amortization)
                            loan_payment_504_ts = build_loan_schedule(
                                loan_504_principal, LOAN_504_ANNUAL_RATE, LOAN_504_TERM_YEARS, IO_MONTHS_504, MONTHS
                            )
                            loan_payment_7a_ts = build_loan_schedule(
                                loan_7a_principal, LOAN_7A_ANNUAL_RATE, LOAN_7A_TERM_YEARS, IO_MONTHS_7A, MONTHS
                            )
                            loan_payment_total_ts = loan_payment_504_ts + loan_payment_7a_ts
    
                            loan_principal_total = loan_504_principal + loan_7a_principal
    
                            cash_balance = 0.0
                            active_members = []   

                            # ---- Staged CapEx: dynamic effects + per-sim queue (mirror main sim) ----
                            _dyn_STATIONS = {k: dict(v) for k, v in STATIONS.items()}
                            _dyn_MAX_MEMBERS = int(MAX_MEMBERS)
                            _dyn_CLAY_COGS_MULT = float(globals().get("CLAY_COGS_MULT", 1.0))
                            _dyn_PUGMILL_MAINT = float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0))
                            _dyn_SLAB_MAINT = float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))

                            _capex_queue = []
                            try:
                                for _it in CAPEX_ITEMS:
                                    unit = float(_it.get("unit_cost", 0.0) or 0.0)
                                    cnt  = int(_it.get("count", 1) or 1)
                                    mth  = _it.get("month", None)
                                    thr  = _it.get("member_threshold", None)
                                    lbl  = _it.get("label", "")
                                    total_cost = (unit * max(1, cnt)) if unit > 0 else float(_it.get("amount", 0.0) or 0.0)
                                    if total_cost > 0 and (mth is not None or thr is not None):
                                        _capex_queue.append({
                                            "unit_cost": unit,
                                            "count": cnt,
                                            "month": mth,
                                            "member_threshold": thr,
                                            "label": lbl,
                                            "purchased": False,
                                        })
                            except Exception:
                                _capex_queue = []

                            remaining_pool = {
                                "community_studio": int(COMMUNITY_POOL),
                                "home_studio":      int(HOME_POOL),
                                "no_access":        int(NO_ACCESS_POOL),
                            }

                            cs_eligible = 0
                            expansion_triggered = False
                            grant_month = scen_cfg["grant_month"]; grant_amount = scen_cfg["grant_amount"]
                            min_cash = float("inf")
                            
                            # --- Workshops state (mirror main sim loop) ---
                            # Build only when enabled; otherwise keep zero series.
                            stream = {}
                            stream["workshop_revenue"] = np.zeros(MONTHS)
                            stream["joins_from_workshops"] = np.zeros(MONTHS, dtype=int)
                            if bool(globals().get("WORKSHOPS_ENABLED", False)):
                                apply_workshops(stream, globals(), MONTHS)
    
                            for month in range(MONTHS):
                                seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                                is_downturn = (rng.random() < DOWNTURN_PROB_PER_MONTH)
                                churn_mult = DOWNTURN_CHURN_MULT if is_downturn else 1.0
                            
                                # ----- Beginner classes (optional, semester-aware) -----
                                revenue_classes_gross = 0.0
                                classes_cost = 0.0
                                class_students_this_month = 0
                                revenue_classes = 0.0
                            
                                if CLASSES_ENABLED and _is_class_month(month):
                                    # stochastic fill around mean
                                    for _ in range(int(CLASS_COHORTS_PER_MONTH)):
                                        fill = rng.normal(CLASS_FILL_MEAN, 0.08)
                                        fill = float(np.clip(fill, 0.0, 1.0))
                                        seats = int(round(CLASS_CAP_PER_COHORT * fill))
                                        class_students_this_month += seats
                                        revenue_classes_gross += seats * CLASS_PRICE
                                        classes_cost += (seats * CLASS_COST_PER_STUDENT) + (CLASS_INSTR_RATE_PER_HR * CLASS_HOURS_PER_COHORT)
                            
                                    # schedule conversion of a fraction of students to members after a lag
                                    # keep a small queue keyed by target month
                                    if month == 0:
                                        pending_class_conversions = {}
                                    target_m = month + int(CLASS_CONV_LAG_MO)
                                    converts = int(round(class_students_this_month * CLASS_CONV_RATE))
                                    if converts > 0:
                                        pending_class_conversions[target_m] = pending_class_conversions.get(target_m, 0) + converts
                            
                                    # Net class revenue (flow into total_revenue)
                                    revenue_classes = max(0.0, revenue_classes_gross - classes_cost)
                            
                                # conversions materialize this month (if any)
                                class_joins_now = 0
                                if CLASSES_ENABLED and 'pending_class_conversions' in locals():
                                    class_joins_now = int(pending_class_conversions.pop(month, 0))
    
                                # Replenish pools each month
                                for _k, _v in MARKET_POOLS_INFLOW.items():
                                    remaining_pool[_k] += int(_v)
    
                                # Unlock CS tranche
                                unlock_now = compute_cs_unlock_share(month, remaining_pool["community_studio"])
                                unlock_now = min(unlock_now, remaining_pool["community_studio"])
                                remaining_pool["community_studio"] -= unlock_now
                                cs_eligible += unlock_now
    
                                # Joins (coarse – same intent structure)
                                noise = rng.lognormal(mean=-(ADOPTION_SIGMA**2)/2, sigma=ADOPTION_SIGMA)
                                price_mult = price_mult_joins
                                cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                                capacity_damping = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
                                intent_common_mult = (seasonal * (DOWNTURN_JOIN_MULT if is_downturn else 1.0) *
                                                      awareness_multiplier(month) * wom_multiplier(len(active_members)) *
                                                      capacity_damping * noise * price_mult)
                                if JOIN_MODEL == "compartment":
                                    lam_comm = max(0.0, BASELINE_RATE_COMMUNITY * intent_common_mult)
                                    pool_intents = {
                                        "community_studio": _haz_to_prob(lam_comm),
                                    }
                                else:
                                    pool_intents = {
                                        "community_studio": POOL_BASE_INTENT["community_studio"] * intent_common_mult,
                                    }
                                
                                joins_no_access   = draw_adopters(remaining_pool["no_access"],   pool_intents["no_access"],rng)
                                joins_home        = draw_adopters(remaining_pool["home_studio"], pool_intents["home_studio"],rng)
                                joins_comm_studio = draw_adopters(cs_eligible,                   pool_intents["community_studio"],rng)
                                remaining_pool["no_access"] -= joins_no_access
                                remaining_pool["home_studio"] -= joins_home
                                cs_eligible -= joins_comm_studio
                                joins = joins_no_access + joins_home + joins_comm_studio
                                joins += int(locals().get("class_joins_now", 0) or 0)
                                # Cap onboarding simply
                                if MAX_ONBOARDINGS_PER_MONTH is not None and joins > MAX_ONBOARDINGS_PER_MONTH:
                                    joins = MAX_ONBOARDINGS_PER_MONTH
    
                                # Add members
                                n_from_class = int(locals().get("class_joins_now", 0) or 0)
                                for i in range(int(joins)):
                                    arch = rng.choice(
                                        list(MEMBER_ARCHETYPES.keys()),
                                        p=[v["prob"] for v in MEMBER_ARCHETYPES.values()]
                                    )
                                    active_members.append({
                                        "type": arch,
                                        "start_month": month,
                                        "monthly_fee": float(price),
                                        "clay_bags": MEMBER_ARCHETYPES[arch]["clay_bags"],
                                        "src": "class" if i < n_from_class else "other",
                                    })
    
                                # Churn
                                kept = []
                                util_over = max(0.0, (len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)) - 1.0)
                                for m in active_members:
                                    p_leave = month_churn_prob(m["type"], tenure_mo=(month - m["start_month"])) * churn_mult * price_mult_churn
                                    p_leave = np.clip(p_leave * (1.0 + UTILIZATION_CHURN_UPLIFT * util_over), 0.0, 0.99)
                                    if rng.random() > p_leave: kept.append(m)
                                active_members = kept
    
                                # Revenues (simplified calc consistent with main code)
                               
                                 # Legacy staged spend removed — table-driven CapEx only                                revenue_membership = sum(m["monthly_fee"] for m in active_members)
                                revenue_clay = 0.0; revenue_firing = 0.0; total_clay_lbs = 0.0
                                for m in active_members:
                                    bags = rng.choice(m["clay_bags"]); revenue_clay += bags * RETAIL_CLAY_PRICE_PER_BAG
                                    clay_lbs = bags * 25; total_clay_lbs += clay_lbs
                                    revenue_firing += (20*3 + max(0, min(20, clay_lbs-20))*4 + max(0, clay_lbs-40)*5) if clay_lbs>20 else clay_lbs*3
                                
                                # Events: net = gross − materials − optional labor
                                revenue_events_gross = 0.0
                                events_cost_materials = 0.0
                                events_cost_labor = 0.0
    
                                events_this_month = int(np.clip(rng.poisson(BASE_EVENTS_PER_MONTH_LAMBDA * seasonal), 0, EVENTS_MAX_PER_MONTH))
                                for _ in range(events_this_month):
                                    attendees = int(rng.choice(ATTENDEES_PER_EVENT_RANGE))
                                    revenue_events_gross += attendees * TICKET_PRICE
    
                                    mug_unit_cost = float(rng.uniform(*EVENT_MUG_COST_RANGE))
                                    events_cost_materials += attendees * (mug_unit_cost + EVENT_CONSUMABLES_PER_PERSON)
                                    events_cost_labor += EVENT_STAFF_RATE_PER_HOUR * EVENT_HOURS_PER_EVENT
    
                                revenue_events = max(0.0, revenue_events_gross - events_cost_materials - events_cost_labor)
                                
                                # Designated artist studios (stochastic monthly occupancy)
                                ds_occupied = int(rng.binomial(DESIGNATED_STUDIO_COUNT, DESIGNATED_STUDIO_BASE_OCCUPANCY)) if DESIGNATED_STUDIO_COUNT > 0 else 0
                                revenue_designated_studios = ds_occupied * DESIGNATED_STUDIO_PRICE
                                # Workshops revenue — use precomputed values from stream    
                                total_revenue = (
                                    revenue_membership
                                    + revenue_clay
                                    + revenue_firing
                                    + revenue_events
                                    + (
                                            float(stream.get("workshop_revenue", np.zeros(MONTHS))[month])
                                            if globals().get("WORKSHOPS_ENABLED", False) else 0.0
                                        )
                                    + revenue_designated_studios
                                    + (0.0 if not CLASSES_ENABLED else revenue_classes)
                                )
    
                                variable_clay_cost = (total_clay_lbs / 25) * WHOLESALE_CLAY_COST_PER_BAG
                                variable_clay_cost *= float(globals().get("CLAY_COGS_MULT", 1.0))
                                water_cost = (total_clay_lbs / 25) * GALLONS_PER_BAG_CLAY * WATER_COST_PER_GALLON
                                
                                firings = max(MIN_FIRINGS_PER_MONTH, min(MAX_FIRINGS_PER_MONTH, round(
                                    BASE_FIRINGS_PER_MONTH * (len(active_members) / max(1, REFERENCE_MEMBERS_FOR_BASE_FIRINGS))
                                )))
    
                                 # Electricity (De-Staged): kiln-2 only if at least 2 kilns purchased
                                kiln2_on = (_dyn_KILN_COUNT >= 2)
    
                                kwh_per_firing = KWH_PER_FIRING_KMT1027 + (KWH_PER_FIRING_KMT1427 if kiln2_on else 0)
                                electricity_cost = firings * kwh_per_firing * COST_PER_KWH
                                
                                
                                monthly_heating_cost = HEATING_COST_WINTER if month % 12 in [10,11,0,1,2,3] else HEATING_COST_SUMMER
    
                                _rent_growth = float(globals().get("RENT_GROWTH_PCT", 0.0))/100
                                _year_index = (month // 12)
                                rent_this_month = fixed_rent * ((1.0 + _rent_growth) ** _year_index)
                                
                                fixed_opex_profit = rent_this_month + INSURANCE_COST + GLAZE_COST_PER_MONTH + monthly_heating_cost
    
                                # Owner draw gating (same as main sim)
                                in_draw_window = in_owner_draw_window(month)
                                within_stipend_quota = (month < OWNER_STIPEND_MONTHS)
                                owner_draw_now = owner_draw if (in_draw_window and within_stipend_quota) else 0.0
    
                                # Staff after threshold
                                staff_cost = STAFF_COST_PER_MONTH if len(active_members) >= STAFF_EXPANSION_THRESHOLD else 0.0
    
                                # Maintenance (randomized, never negative)
                                maintenance_cost = MAINTENANCE_BASE_COST + max(0, rng.normal(0, MAINTENANCE_RANDOM_STD))
                                maintenance_cost += float(globals().get("PUGMILL_MAINT_COST_PER_MONTH", 0.0)) + float(globals().get("SLAB_ROLLER_MAINT_COST_PER_MONTH", 0.0))
    
                                # Marketing ramp
                                marketing_cost = MARKETING_COST_BASE * (MARKETING_RAMP_MULTIPLIER if month < MARKETING_RAMP_MONTHS else 1.0)
    
                                # S-corp owner salary + payroll taxes (match main sim)
                                owner_salary_expense = 0.0
                                employer_payroll_tax = 0.0
                                employee_withholding = 0.0
                                if ENTITY_TYPE == "s_corp":
                                    owner_salary_expense = SCORP_OWNER_SALARY_PER_MONTH
                                    employer_payroll_tax = owner_salary_expense * EMPLOYER_PAYROLL_TAX_RATE
                                    employee_withholding = owner_salary_expense * EMPLOYEE_PAYROLL_TAX_RATE
    
                                fixed_opex_cash = fixed_opex_profit + loan_payment_total_ts[month] + owner_draw_now
                                total_opex_cash = (
                                    fixed_opex_cash
                                    + variable_clay_cost
                                    + water_cost
                                    + electricity_cost
                                    + staff_cost
                                    + maintenance_cost
                                    + marketing_cost
                                    + owner_salary_expense
                                    + employer_payroll_tax
                                    + employee_withholding
                                )
                                
                                # --- Lightweight taxes & remittances (quarterly) + sales tax + property tax ---
                                if month == 0:
                                    # accumulators that persist across months within this sim
                                    se_ss_wage_base_used_ytd = 0.0
                                    se_tax_payable_accum = 0.0
                                    state_tax_payable_accum = 0.0
                                    corp_tax_payable_accum = 0.0
                                    sales_tax_payable_accum = 0.0
                                    
                                # Reset SE wage base every January (match main sim)
                                if (month % 12) == 0 and month > 0:
                                    se_ss_wage_base_used_ytd = 0.0
    
                                # Approximate operating profit (pre-debt, pre-draw; matches main sim’s tax base)
                                op_profit_approx = (
                                    total_revenue
                                    - (fixed_opex_profit
                                       + variable_clay_cost + water_cost + electricity_cost
                                       + staff_cost + maintenance_cost + marketing_cost
                                       + owner_salary_expense + employer_payroll_tax)
                                )
    
                                se_tax_this_month = 0.0
                                state_income_tax_this_month = 0.0
                                corp_tax_this_month = 0.0
    
                                if ENTITY_TYPE in ("sole_prop", "partnership"):
                                    se_earnings = max(0.0, op_profit_approx) * SE_EARNINGS_FACTOR
                                    ss_base_remaining = max(0.0, SE_SOC_SEC_WAGE_BASE - se_ss_wage_base_used_ytd)
                                    ss_taxable_now = min(se_earnings, ss_base_remaining)
                                    se_tax_ss = ss_taxable_now * SE_SOC_SEC_RATE
                                    se_ss_wage_base_used_ytd += ss_taxable_now
                                    se_tax_medicare = se_earnings * SE_MEDICARE_RATE
                                    se_tax_this_month = se_tax_ss + se_tax_medicare
                                    se_tax_payable_accum += se_tax_this_month
    
                                    half_se_deduction = 0.5 * se_tax_this_month
                                    ma_taxable_income = max(0.0, op_profit_approx - half_se_deduction)
                                    state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                                    state_tax_payable_accum += state_income_tax_this_month
    
                                elif ENTITY_TYPE == "s_corp":
                                    ma_taxable_income = max(0.0, op_profit_approx)
                                    state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                                    state_tax_payable_accum += state_income_tax_this_month
    
                                elif ENTITY_TYPE == "c_corp":
                                    corp_taxable_income = max(0.0, op_profit_approx)
                                    corp_tax_this_month = corp_taxable_income * (FED_CORP_TAX_RATE + MA_CORP_TAX_RATE)
                                    corp_tax_payable_accum += corp_tax_this_month
    
                                # Sales tax on clay retail (cash collected → remit quarterly)
                                sales_tax_collected = revenue_clay * MA_SALES_TAX_RATE
                                sales_tax_payable_accum += sales_tax_collected
    
                                # Quarterly remittances (cash)
                                tax_payments_this_month = 0.0
                                if ((month + 1) % ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                                    if ENTITY_TYPE in ("sole_prop", "partnership", "s_corp"):
                                        tax_payments_this_month += se_tax_payable_accum
                                        tax_payments_this_month += state_tax_payable_accum
                                        se_tax_payable_accum = 0.0
                                        state_tax_payable_accum = 0.0
                                    if ENTITY_TYPE == "c_corp":
                                        tax_payments_this_month += corp_tax_payable_accum
                                        corp_tax_payable_accum = 0.0
    
                                if ((month + 1) % SALES_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                                    tax_payments_this_month += sales_tax_payable_accum
                                    sales_tax_payable_accum = 0.0
    
                                # Annual personal property tax (cash only)
                                if PERSONAL_PROPERTY_TAX_ANNUAL > 0 and ((month + 1) % 12 == (PERSONAL_PROPERTY_TAX_BILL_MONTH % 12)):
                                    tax_payments_this_month += PERSONAL_PROPERTY_TAX_ANNUAL
    
                                # Add tax remittances to cash OpEx
                                total_opex_cash += tax_payments_this_month
    
                                # Month 0 loan proceeds / CapEx / cash-paid fees (match main sim; use total of 504 + 7(a))

                                if month == 0:
                                    if LOAN_MODE == "upfront":
                                        upfront_capex = 0
                                        cash_balance += loan_principal_total - upfront_capex - (fees_cash_outflow if 'fees_cash_outflow' in locals() else 0.0)
                                    else:
                                        # staged: no proceeds at t=0 in lightweight path either
                                        pass
                                
                                # ---- Staged CapEx purchases (month- or membership-triggered) ----
                                capex_draw_this_month = 0.0
                                if _capex_queue:
                                    current_members = len(active_members)
                                    for _item in _capex_queue:
                                        if _item["purchased"]:
                                            continue
                                        m_ok = (_item["month"] is not None) and (month == int(_item["month"]))
                                        n_ok = (_item["member_threshold"] is not None) and (current_members >= int(_item["member_threshold"]))
                                        if m_ok or n_ok:
                                            cnt = int(_item.get("count", 1) or 1)
                                            unit = float(_item.get("unit_cost", 0.0) or 0.0)
                                            total_cost = unit * cnt
                                            capex_draw_this_month += total_cost
                                            # ---- Apply equipment effects based on label ----
                                            lbl = str(_item.get("label", "")).lower()
                                           # Wheels: TOTAL wheels target
                                            if "wheel" in lbl:
                                                if "kiln" in lbl:
                                                    _dyn_KILN_COUNT += max(1, cnt)
                                                if "wheels" in _dyn_STATIONS:
                                                    curr = int(_dyn_STATIONS["wheels"].get("capacity", 0))
                                                    target = max(0, cnt)
                                                    _dyn_STATIONS["wheels"]["capacity"] = max(curr, target)
                                            # Wire racks: TOTAL racks target
                                            if "rack" in lbl:
                                                target_members = 3 * max(0, cnt)
                                                _dyn_MAX_MEMBERS = max(_dyn_MAX_MEMBERS, target_members)
                                            if "slab" in lbl and "roll" in lbl:
                                                if "handbuilding" in _dyn_STATIONS:
                                                    hb = int(_dyn_STATIONS["handbuilding"].get("capacity", 6))
                                                    _dyn_STATIONS["handbuilding"]["capacity"] = max(1, int(round(hb * 1.20)))
                                                _dyn_SLAB_MAINT += 10.0
                                            if "pug" in lbl:
                                                _dyn_CLAY_COGS_MULT = 0.75
                                                _dyn_PUGMILL_MAINT += 20.0
                                            _item["purchased"] = True
                                    if capex_draw_this_month > 0.0:
                                        cash_balance -= capex_draw_this_month
                                        # If staged CapEx: draw tranche into the 504 schedule and start payments next month
                                        if CAPEX_MODE == "staged":
                                            draw_pct = float(LOAN_STAGED_RULE.get("draw_pct_of_purchase", 1.0))
                                            tranche = capex_draw_this_month * max(0.0, min(draw_pct, 1.0))
                                            tranche = max(tranche, float(LOAN_STAGED_RULE.get("min_tranche", 0.0)))
                                            mx = LOAN_STAGED_RULE.get("max_tranche", None)
                                            if mx is not None:
                                                tranche = min(tranche, float(mx))
                                            if tranche > 0.0:
                                                cash_balance += tranche
                                                loan_tranche_draw_capex = tranche  # <-- expose per-month 504 draw
                                                _add_staged_tranche_into_array(
                                                    loan_payment_504_ts,  # <-- write into the 504 schedule
                                                    month,
                                                    tranche,
                                                    LOAN_504_ANNUAL_RATE,  # <-- 504 rate
                                                    LOAN_504_TERM_YEARS,   # <-- 504 term
                                                    IO_MONTHS_504,         # <-- 504 IO months
                                                    MONTHS
                                                )
                                                # keep 'total' in sync after changing a component schedule
                                                loan_payment_total_ts = loan_payment_504_ts + loan_payment_7a_ts

                                        
                                        STATIONS.update(_dyn_STATIONS)
                                        globals()["MAX_MEMBERS"] = int(_dyn_MAX_MEMBERS)
                                        globals()["CLAY_COGS_MULT"] = float(_dyn_CLAY_COGS_MULT)
                                        globals()["PUGMILL_MAINT_COST_PER_MONTH"] = float(_dyn_PUGMILL_MAINT)
                                        globals()["SLAB_ROLLER_MAINT_COST_PER_MONTH"] = float(_dyn_SLAB_MAINT)

                                cash_balance += (total_revenue - total_opex_cash)
                                if (grant_month is not None) and (month == grant_month):
                                    cash_balance += grant_amount
                                min_cash = min(min_cash, cash_balance)
    
                            rows_min.append({"scenario": scen_name, "rent": fixed_rent, "owner_draw": owner_draw, "min_cash": min_cash})
    
            df = pd.DataFrame(rows_min)
            agg = df.groupby(["scenario","rent","owner_draw"])["min_cash"].median().reset_index()
            agg["extra_buffer"] = b
            out.append(agg)
    
        EXTRA_BUFFER = baseline_extra
        return pd.concat(out, ignore_index=True)
    
    # =============================================================================
    # Business-plan dashboard (plots + lender summary)
    # =============================================================================
    def plot_business_plan_dashboard(
        results_df: pd.DataFrame,
        scenario: str,
        rent: float,
        owner_draw: float,
        membership_soft_cap: float,
        max_members: Optional[int] = None,  # ✅ fixed type hint
        dscr_target: float = 1.25
    ):
        """
        Creates a dashboard of 8 charts + a lender summary for a given configuration.
        Assumes results_df has columns created by the simulation above.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        sns.set_context("talk")
        sns.set_style("whitegrid")
    
        cfg = results_df.query(
            "scenario == @scenario and rent == @rent and owner_draw == @owner_draw"
        ).copy()
        if cfg.empty:
            print("No rows match the requested configuration.")
            return
    
        # --- Common aggregates
        months = np.arange(1, cfg["month"].max() + 1)
        # Revenue components and totals
        # Revenue components and totals
        rev_cols = [
            "revenue_membership",
            "revenue_firing",
            "revenue_clay",
            "revenue_events",
            "revenue_workshops_net",
            "revenue_designated_studios",
            "revenue_classes",  # NEW: classes (net)
        ]
        cfg["rev_total"] = cfg[rev_cols].sum(axis=1)
        cfg["opex_cash"] = cfg["rev_total"] - cfg["net_cash_flow"]
    
        # Median bands helper
        def band(series):
            g = series.groupby(cfg["month"])
            return g.median(), g.quantile(0.10), g.quantile(0.90)
    
        # --- 1) Cash runway (median + 10–90%)
        med, p10, p90 = band(cfg["cash_balance"])
        plt.figure(figsize=(10, 6))
        plt.plot(months, med.reindex(months).values, linewidth=2, label="Median")
        plt.fill_between(months, p10.reindex(months).values, p90.reindex(months).values, alpha=0.12, label="10–90%")
        # Grant markers
        grant_info = cfg[["grant_month", "grant_amount"]].drop_duplicates()
        for _, row in grant_info.iterrows():
            gm, ga = row["grant_month"], row["grant_amount"]
            if pd.notna(gm) and gm is not None and ga and ga > 0:
                plt.axvline(gm + 1, color="green", linestyle="--", linewidth=1.5)
                ymin, ymax = plt.ylim()
                plt.text(gm + 1, ymin + 0.05 * (ymax - ymin), f"Grant ${ga:,.0f}", color="green", rotation=90, va="bottom")
        plt.axhline(0, color="black", linestyle=":", linewidth=1)
        plt.title(f"Cash Balance — {scenario} | Rent ${rent:,.0f} | Draw ${owner_draw:,.0f}")
        plt.xlabel("Month"); plt.ylabel("Cash ($)")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 2) Cash-at-risk curve (P[cash<0] by month)
        car = cfg.groupby("month")["cash_balance"].apply(lambda s: (s < 0).mean())
        plt.figure(figsize=(10, 4.5))
        plt.plot(car.index, car.values, linewidth=2)
        plt.ylim(0, 1)
        plt.title("Cash-at-Risk: Probability(Cash < $0) by Month")
        plt.xlabel("Month"); plt.ylabel("Probability")
        plt.tight_layout(); plt.show()
    
        # --- 3) Operating break-even ECDF (P(BE <= t))
        be_by_sim = (
            cfg.sort_values(["simulation_id", "month"])
               .groupby("simulation_id")
               .apply(lambda d: d.loc[d["cumulative_op_profit"] >= 0, "month"].min())
               .dropna()
        )
        ecdf = pd.Series({t: (be_by_sim <= t).mean() for t in months})
        plt.figure(figsize=(10, 4.5))
        plt.plot(ecdf.index, ecdf.values, linewidth=2)
        if not be_by_sim.empty:
            be_median = be_by_sim.median()
            plt.axvline(be_median, linestyle="--", linewidth=1.5, color="gray")
            plt.text(be_median, 0.05, f"Median BE ≈ M{int(be_median)}", rotation=90, va="bottom", ha="right")
        plt.ylim(0, 1)
        plt.title("Operating Break-even ECDF")
        plt.xlabel("Month"); plt.ylabel("P(BE reached by month)")
        plt.tight_layout(); plt.show()
    
        # --- 4) DSCRs over time (Op‑Profit DSCR and Cash‑DSCR) with target
        dscr_op = cfg["dscr"].replace([np.inf, -np.inf], np.nan)
        dscr_cash_series = cfg["dscr_cash"].replace([np.inf, -np.inf], np.nan)
    
        op_med, op_p10, op_p90 = band(dscr_op)
        cash_med, cash_p10, cash_p90 = band(dscr_cash_series)
    
        plt.figure(figsize=(10, 4.8))
        # Op‑Profit DSCR band
        plt.plot(months, op_med.reindex(months).values, linewidth=2, label="Op‑Profit DSCR (median)")
        plt.fill_between(months, op_p10.reindex(months).values, op_p90.reindex(months).values, alpha=0.10, label="Op‑Profit DSCR (10–90%)")
        # Cash‑DSCR band
        plt.plot(months, cash_med.reindex(months).values, linewidth=2, linestyle="--", label="Cash‑DSCR (median)")
        plt.fill_between(
        months,
        cash_p10.reindex(months).values,
        cash_p90.reindex(months).values,
        alpha=0.10,
        label="Cash‑DSCR (10–90%)"
    )
        
        
        # Target line
        plt.axhline(dscr_target, linestyle="--", linewidth=1.5, label=f"Target {dscr_target:.2f}×")
        plt.title("DSCR Over Time — Operating vs Cash")
        plt.xlabel("Month"); plt.ylabel("DSCR (×)")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 5) Membership vs caps
        m_med, m_p10, m_p90 = band(cfg["active_members"])
        plt.figure(figsize=(10, 4.5))
        plt.plot(months, m_med.reindex(months).values, linewidth=2, label="Median")
        plt.fill_between(months, m_p10.reindex(months).values, m_p90.reindex(months).values, alpha=0.12, label="10–90%")
        plt.axhline(membership_soft_cap, linestyle="--", linewidth=1.5, label=f"Soft cap ≈ {membership_soft_cap:.0f}")
        if max_members is not None:
            plt.axhline(max_members, linestyle=":", linewidth=1.5, color="orange", label=f"Hard cap = {max_members}")
        plt.title("Active Members — Capacity View")
        plt.xlabel("Month"); plt.ylabel("Members")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 6) Revenue mix (median) vs OpEx (cash) & Net Cash Flow
        g_rev = cfg.groupby("month")[rev_cols].median()
        g_ops = cfg.groupby("month")[["rev_total", "opex_cash", "net_cash_flow"]].median()
        plt.figure(figsize=(11, 5.5))
        plt.stackplot(
            g_rev.index,
            g_rev["revenue_membership"],
            g_rev["revenue_firing"],
            g_rev["revenue_clay"],
            g_rev["revenue_events"],
            g_rev["revenue_workshops_net"],
            g_rev["revenue_designated_studios"],
            g_rev["revenue_classes"],  # NEW
            labels=[
                "Membership", "Firing", "Clay", "Events",
                "Workshops (net)", "Designated studios", "Classes (net)"  # NEW
            ],
            alpha=0.9,
        )
        
        plt.plot(g_ops.index, g_ops["opex_cash"], linewidth=2.0, label="Total OpEx (cash)")
        plt.plot(g_ops.index, g_ops["net_cash_flow"], linestyle="--", linewidth=1.5, label="Net cash flow")
        plt.title("Revenue Composition vs Cash OpEx")
        plt.xlabel("Month"); plt.ylabel("Dollars ($)")
        plt.legend(loc="upper center", ncol=3)
        plt.tight_layout(); plt.show()
    
        # --- 7) Unit metrics: Revenue per Member & Net CF per Member (median)
        unit = cfg.copy()
        unit["rev_per_member"] = unit["rev_total"] / unit["active_members"].clip(lower=1)
        unit["ncf_per_member"] = unit["net_cash_flow"] / unit["active_members"].clip(lower=1)
        up = unit.groupby("month")[["rev_per_member", "ncf_per_member"]].median()
        plt.figure(figsize=(10, 4.5))
        plt.plot(up.index, up["rev_per_member"], linewidth=2, label="Revenue / Member / Month")
        plt.plot(up.index, up["ncf_per_member"], linewidth=2, linestyle="--", label="Net Cash Flow / Member / Month")
        plt.title("Unit Economics (Median)")
        plt.xlabel("Month"); plt.ylabel("$/Member/Month")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 8) Stress lens: Min cash distribution across simulations
        min_cash_by_sim = cfg.groupby("simulation_id")["cash_balance"].min()
        plt.figure(figsize=(10, 4.5))
        bins = max(10, min(40, int(np.sqrt(len(min_cash_by_sim)))))
        plt.hist(min_cash_by_sim.values, bins=bins, alpha=0.6)
        plt.axvline(min_cash_by_sim.median(), color="black", linestyle="--", linewidth=1.5, label=f"Median = ${min_cash_by_sim.median():,.0f}")
        plt.title("Minimum Cash Across Simulations")
        plt.xlabel("Minimum Cash over 60 Months ($)"); plt.ylabel("Count")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- Lender summary (concise)
        # Key stats at months 12 and 24
        def pct(x): return f"{100*x:.0f}%"
        car_12 = (cfg.loc[cfg["month"] == 12, "cash_balance"] < 0).mean() if (cfg["month"] == 12).any() else np.nan
        car_24 = (cfg.loc[cfg["month"] == 24, "cash_balance"] < 0).mean() if (cfg["month"] == 24).any() else np.nan
        dscr_12 = cfg.loc[cfg["month"] == 12, "dscr"].median() if (cfg["month"] == 12).any() else np.nan
        dscr_24 = cfg.loc[cfg["month"] == 24, "dscr"].median() if (cfg["month"] == 24).any() else np.nan
        dscr_cash_12 = cfg.loc[cfg["month"] == 12, "dscr_cash"].median() if (cfg["month"] == 12).any() else np.nan
        dscr_cash_24 = cfg.loc[cfg["month"] == 24, "dscr_cash"].median() if (cfg["month"] == 24).any() else np.nan
        be_m = be_by_sim.median() if not be_by_sim.empty else np.nan
        insol_before_grant = cfg.groupby("simulation_id")["insolvent_before_grant"].max().mean()
    
        # Owner take-home (if table exists)
        try:
            oth = summarize_owner_takehome(results_df)
            oth_row = oth.query("scenario == @scenario and rent == @rent and owner_draw == @owner_draw")
            owner_takehome_median = float(oth_row["median_owner_takehome_annual"].iloc[0]) if not oth_row.empty else np.nan
        except Exception:
            owner_takehome_median = np.nan
    
        lender_summary = pd.DataFrame({
            "Scenario": [scenario],
            "Rent ($/mo)": [f"{rent:,.0f}"],
            "Owner draw ($/mo)": [f"{owner_draw:,.0f}"],
            "Median BE month": [None if pd.isna(be_m) else int(be_m)],
            "CAR @12": [None if pd.isna(car_12) else pct(car_12)],
            "CAR @24": [None if pd.isna(car_24) else pct(car_24)],
            "DSCR @12 (Op‑Profit, p50)": [None if pd.isna(dscr_12) else f"{dscr_12:.2f}x"],
            "DSCR @24 (Op‑Profit, p50)": [None if pd.isna(dscr_24) else f"{dscr_24:.2f}x"],
            "DSCR @12 (Cash, p50)": [None if pd.isna(dscr_cash_12) else f"{dscr_cash_12:.2f}x"],
            "DSCR @24 (Cash, p50)": [None if pd.isna(dscr_cash_24) else f"{dscr_cash_24:.2f}x"],
            "Median min cash ($)": [f"{min_cash_by_sim.median():,.0f}"],
            "Median cash @M60 ($)": [f"{cfg.loc[cfg['month']==cfg['month'].max(), 'cash_balance'].median():,.0f}"],
            "% sims insolvent pre-grant": [pct(insol_before_grant)],
            "Owner take-home (annual, median)": [None if pd.isna(owner_takehome_median) else f"${owner_takehome_median:,.0f}"],
        })
        print("\n=== Lender Summary (focus configuration) ===")
        print(lender_summary.to_string(index=False))
    
    
    # ---- Example call (choose one configuration to render) ----
    # Update these three to the combo you want to present:
    scenario_focus   = "Base"
    rent_focus       = 3500
    owner_draw_focus = 0
    
    # Try to pass MAX_MEMBERS if defined, else None
    try:
        max_members_focus = int(MAX_MEMBERS)
    except Exception:
        max_members_focus = None
    
    plot_business_plan_dashboard(
        results_df=results_df,
        scenario=scenario_focus,
        rent=rent_focus,
        owner_draw=owner_draw_focus,
        membership_soft_cap=MEMBERSHIP_SOFT_CAP,
        max_members=max_members_focus,
        dscr_target=1.25
    )
    # At the very end, return the artifacts you care about:
    return {
        "results_df": results_df,
        "summary_table": summary_table,
        "owner_takehome_table": owner_takehome_table
    }


def run_from_cfg(cfg: dict | None = None):
    """
    Public entrypoint for batch/adapter use.
    - Merges cfg with defaults captured from this file
    - Temporarily overrides module constants
    - Runs your original code path unchanged
    - Restores globals afterward
    Returns a dict with DataFrames, and your existing side‑effects (plots/CSVs/prints) still happen.
    """
    merged = resolve_cfg(cfg)

    # Inject downturn prob (unless caller pinned it)
    p, p_src = _get_downturn_prob(merged)
    merged.setdefault("DOWNTURN_PROB_PER_MONTH", p)
    print(f"[nowcast] DOWNTURN_PROB_PER_MONTH = {merged['DOWNTURN_PROB_PER_MONTH']:.3f}  (source={p_src})")

    with override_globals(merged):
        # If any helpers previously read globals, they still will—now pointed at merged.
        artifacts = _core_simulation_and_reports()
    
    # OPTIONAL: stamp the value/source into returned tables for traceability
    try:
        artifacts["summary_table"]["downturn_prob_per_month"] = merged["DOWNTURN_PROB_PER_MONTH"]
        artifacts["summary_table"]["downturn_source"] = p_src
    except Exception:
        pass


    return artifacts

if __name__ == "__main__":
    # Running the file directly behaves exactly as before:
    run_from_cfg({})
