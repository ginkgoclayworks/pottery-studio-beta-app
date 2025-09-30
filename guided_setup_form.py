#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guided_setup_form.py
One-page Guided Setup form for ceramics studio financial simulator.
"""

from typing import Dict, Any, List
import streamlit as st


# ---------------- helpers ----------------
def _get(ps: Dict[str, Any], key: str, default):
    """Read key from params_state with fallback."""
    return ps.get(key, default)


def _capex_pack_template(pack: str) -> List[Dict[str, Any]]:
    """
    Simple preset equipment packs.
    Names should match existing CAPEX_ITEMS rows in your app.
    """
    catalog = [
        {"name": "Electric Kiln (L&L/Skutt)", "unit_cost": 5500, "qty": 1},
        {"name": "Pottery Wheels",            "unit_cost": 1200, "qty": 4},
        {"name": "Slab Roller",               "unit_cost": 1800, "qty": 1},
        {"name": "Wire Shelving Racks",       "unit_cost": 110,  "qty": 10},
        {"name": "Pugmill",                   "unit_cost": 5500, "qty": 0},
        {"name": "Ventilation / Envirovent",  "unit_cost": 700,  "qty": 1},
        {"name": "Glaze Table & Buckets",     "unit_cost": 600,  "qty": 1},
        {"name": "Electrical Upgrade",        "unit_cost": 4000, "qty": 0},
        {"name": "Plumbing / Sink & Trap",    "unit_cost": 2500, "qty": 0},
    ]
    packs = {
        "Starter": {"Electric Kiln (L&L/Skutt)":1,"Pottery Wheels":4,"Slab Roller":1,
                    "Wire Shelving Racks":10,"Ventilation / Envirovent":1,"Glaze Table & Buckets":1},
        "Growth":  {"Electric Kiln (L&L/Skutt)":1,"Pottery Wheels":6,"Slab Roller":1,
                    "Wire Shelving Racks":14,"Ventilation / Envirovent":1,"Glaze Table & Buckets":1,
                    "Pugmill":1,"Electrical Upgrade":1,"Plumbing / Sink & Trap":1},
        "Full":    {"Electric Kiln (L&L/Skutt)":2,"Pottery Wheels":8,"Slab Roller":1,
                    "Wire Shelving Racks":18,"Ventilation / Envirovent":1,"Glaze Table & Buckets":1,
                    "Pugmill":1,"Electrical Upgrade":1,"Plumbing / Sink & Trap":1},
    }
    target = packs.get(pack, packs["Starter"])
    out = []
    for item in catalog:
        name = item["name"]
        qty = target.get(name, item["qty"])
        out.append({
            "name": name,
            "unit_cost": item["unit_cost"],
            "qty": qty,
            "enabled": qty > 0,
            "finance_504": qty > 0,
        })
    return out


def _merge_capex(existing: List[Dict[str, Any]] | None,
                 templ: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge CAPEX template into existing CAPEX_ITEMS by name.
    Preserve existing unit_cost if user customized it.
    """
    existing = existing or []
    by_name = {x.get("name"): dict(x) for x in existing if x.get("name")}
    for t in templ:
        n = t["name"]
        if n in by_name:
            prev = by_name[n]
            unit_cost = prev.get("unit_cost", t["unit_cost"])
            prev.update({**t, "unit_cost": unit_cost})
            by_name[n] = prev
        else:
            by_name[n] = t
    # preserve stable order
    seen, merged = set(), []
    for x in existing:
        nm = x.get("name")
        if nm in by_name and nm not in seen:
            merged.append(by_name[nm]); seen.add(nm)
    for nm, x in by_name.items():
        if nm not in seen:
            merged.append(x)
    return merged


# ---------------- public entry ----------------
def guided_setup_form(params_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render a single-page Guided Setup form.
    Prefills from params_state, writes back on submit.
    """
    with st.form("guided_setup_form", clear_on_submit=False):
        st.subheader("Guided Setup (one page)")
        st.caption("Defaults are pre-filled from your current model.")
        st.info(
            "This page sets the **big knobs** for your studio. "
            "**504** is for equipment/build-out (CapEx). **7(a)** is for operating runway (OpEx). "
            "You can refine details later in the Advanced sections."
        )

        # ---- Space & Rent ----
        c1, c2 = st.columns(2)
        with c1:
            space_sqft = st.number_input(
                "Studio size (sq ft)", 200, 20000,
                int(_get(params_state, "SPACE_SQFT", 2000)), step=50,
                help="Total usable space for members/classes. Used for capacity planning and cost per sq ft views."
            )
        with c2:
            rent = st.number_input(
                "Monthly rent ($/mo)", 200, 50000,
                int(_get(params_state, "RENT", 3000)), step=100,
                help="Base monthly rent. If you don't model utilities/CAM separately yet, include them here."
            )

        # ---- Membership ----
        c3, c4 = st.columns(2)
        with c3:
            max_members = st.number_input(
                "Max members (capacity)", 1, 500,
                int(_get(params_state, "MAX_MEMBERS", 30)), step=5,
                help="Upper limit of active members your studio can support. The simulator won't exceed this ceiling."
            )
        with c4:
            member_fee = st.number_input(
                "Membership fee ($/month)", 20.0, 1000.0,
                float(_get(params_state, "PRICE", 225.0)),
                step=5.0, 
                help="Monthly price charged per member. This is the PRICE parameter in the simulator."
            )

        # ---- Equipment ----
        equip_pack = st.selectbox(
            "Equipment pack",
            ["Starter","Growth","Full"],
            index=["Starter","Growth","Full"].index(
                params_state.get("EQUIP_PACK", "Starter")
            ),
            help="Sets quantities for wheels, kilns, racks, etc. These flow into CapEx and are 504-financeable by default. You can fine-tune later."
        )

        # ---- Owner draw ----
        owner_draw = st.number_input(
            "Owner draw ($/mo)", 0, 20000,
            int(_get(params_state, "OWNER_DRAW", 0)), step=100,
            help="Monthly pay you take from the business. Affects cash flow and DSCR."
        )

        # ---- Events ----
        st.markdown("**Event Pricing & Costs**")
        st.caption("For paint-a-pot / sip-and-paint style events")
        
        c5, c6 = st.columns(2)
        with c5:
            events_per_month = st.number_input(
                "Events per month", 0, 20,
                int(_get(params_state, "BASE_EVENTS_PER_MONTH_LAMBDA", 2)), step=1,
                help="Average events hosted each month (e.g., Make-Your-Own-Mug). Uses Poisson distribution with seasonality."
            )
        with c6:
            event_price = st.number_input(
                "Ticket price per person ($)", 20, 500,
                int(_get(params_state, "TICKET_PRICE", 85)), step=5,
                help="Price charged per event attendee."
            )
        
        c7, c8 = st.columns(2)
        with c7:
            event_size = st.number_input(
                "Attendees per event", 4, 24,
                int(_get(params_state, "ATTENDEES_PER_EVENT_RANGE", [8,12])[1]), step=1,
                help="Typical group size you plan for. Model will randomly vary around this."
            )
        with c8:
            event_consumables = st.number_input(
                "Consumables per attendee ($)", 0, 100,
                int(_get(params_state, "EVENT_CONSUMABLES_PER_PERSON", 10)), step=1,
                help="Cost per person PER EVENT: glazes, brushes, sponges, paper towels, packaging, cleanup materials. Does NOT include bisque mug cost (tracked separately)."
            )

        event_staff_hours = st.number_input(
            "Staff hours per event", 0, 16,
            int(_get(params_state, "EVENT_HOURS_PER_EVENT", 3)), step=1,
            help="Total paid staff time for ONE event including setup, teaching, and cleanup."
        )

        # ---- Loans ----
        st.markdown("**SBA Loan Terms**")
        
        c9, c10 = st.columns(2)
        with c9:
            term_504 = st.selectbox(
                "504 loan term (years)", [5, 7, 10],
                index=[5,7,10].index(int(_get(params_state, "LOAN_504_TERM_YEARS", 7))),
                help="SBA 504 funds **CapEx** (equipment/build-out). Longer terms lower the monthly payment."
            )
        with c10:
            io_504 = st.number_input(
                "504 interest-only months", 0, 24,
                int(_get(params_state, "IO_MONTHS_504", 6)), step=1,
                help="Months at the start where you pay **interest only**. Lowers early cash burn; increases total interest paid."
            )
        
        c11, c12 = st.columns(2)
        with c11:
            term_7a = st.selectbox(
                "7(a) loan term (years)", [5, 7, 10, 15, 25],
                index=[5,7,10,15,25].index(int(_get(params_state, "LOAN_7A_TERM_YEARS", 10))),
                help="SBA 7(a) funds **OpEx runway**. Longer terms lower the monthly payment."
            )
        with c12:
            io_7a = st.number_input(
                "7(a) interest-only months", 0, 24,
                int(_get(params_state, "IO_MONTHS_7A", 6)), step=1,
                help="Months at the start where you pay **interest only** for 7(a). Helps early cash flow; increases total interest."
            )
        
        runway_months = st.number_input(
            "Operating runway months (7a sizing)", 0, 24,
            int(_get(params_state, "RUNWAY_MONTHS", 6)), step=1,
            help="How many months of operating expenses you want 7(a) to cover. **Directly scales the 7(a) loan size.**"
        )

        submitted = st.form_submit_button("Submit Guided Setup", type="primary")

    # ---- Apply values on submit ----
    if submitted:
        ps = dict(params_state)

        # Space & Rent - note SPACE_SQFT may not be used in simulator, but keep for UI consistency
        ps["SPACE_SQFT"] = int(space_sqft)
        ps["RENT"] = int(rent)

        # Membership
        ps["MAX_MEMBERS"] = int(max_members)
        ps["PRICE"] = float(member_fee)  # This is the actual simulator parameter

        # Owner draw
        ps["OWNER_DRAW"] = int(owner_draw)

        # Events - map to correct simulator parameters
        ps["BASE_EVENTS_PER_MONTH_LAMBDA"] = float(events_per_month)
        ps["TICKET_PRICE"] = float(event_price)
        
        # Event size - simulator expects a list/range, so create one
        ps["ATTENDEES_PER_EVENT_RANGE"] = [event_size - 2, event_size, event_size + 2]
        
        # Event consumables - correct parameter name
        ps["EVENT_CONSUMABLES_PER_PERSON"] = float(event_consumables)
        ps["EVENT_HOURS_PER_EVENT"] = float(event_staff_hours)

        # Loan terms
        ps["LOAN_504_TERM_YEARS"] = int(term_504)
        ps["IO_MONTHS_504"] = int(io_504)
        ps["LOAN_7A_TERM_YEARS"] = int(term_7a)
        ps["IO_MONTHS_7A"] = int(io_7a)
        ps["RUNWAY_MONTHS"] = int(runway_months)

        # Equipment pack
        ps["EQUIP_PACK"] = equip_pack
        templ = _capex_pack_template(equip_pack)
        ps["CAPEX_ITEMS"] = _merge_capex(ps.get("CAPEX_ITEMS"), templ)

        # Flag for auto-run
        ps["_guided_setup_complete"] = True

        st.success("Guided setup applied. Running simulation...")
        return ps

    return params_state