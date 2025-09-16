#!/usr/bin/env python3
"""
config_manager.py

Consolidates scattered parameters into logical groups and provides mapping
to the existing simulator's expected format. This serves as the bridge between
a simplified beta UI and the existing complex parameter structure.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json


@dataclass
class BusinessFundamentals:
    """Core operational parameters that define the studio"""
    monthly_rent: float = 4500.0
    max_members: int = 120
    membership_price: float = 85.0
    owner_monthly_compensation: float = 2000.0
    
    def validate(self) -> List[str]:
        errors = []
        if self.monthly_rent <= 0:
            errors.append("Monthly rent must be positive")
        if self.max_members <= 0:
            errors.append("Maximum members must be positive")
        if self.membership_price <= 0:
            errors.append("Membership price must be positive")
        if self.owner_monthly_compensation < 0:
            errors.append("Owner compensation cannot be negative")
        return errors


@dataclass
class FinancingConfig:
    """SBA loan terms and financing structure"""
    # SBA 504 Loan
    loan_504_amount: float = 350000.0
    loan_504_rate: float = 0.065
    loan_504_term_years: int = 20
    loan_504_io_months: int = 6
    
    # SBA 7(a) Loan  
    loan_7a_amount: float = 150000.0
    loan_7a_rate: float = 0.075
    loan_7a_term_years: int = 10
    loan_7a_io_months: int = 6
    
    # Working capital and reserves
    working_capital_target: float = 50000.0
    reserve_floor: float = 10000.0
    runway_months: int = 18
    loan_contingency_pct: float = 0.10
    extra_buffer: float = 25000.0
    
    def validate(self) -> List[str]:
        errors = []
        if self.loan_504_rate <= 0 or self.loan_504_rate > 0.20:
            errors.append("SBA 504 rate should be between 0% and 20%")
        if self.loan_7a_rate <= 0 or self.loan_7a_rate > 0.20:
            errors.append("SBA 7(a) rate should be between 0% and 20%")
        if self.loan_504_term_years <= 0:
            errors.append("SBA 504 term must be positive")
        if self.loan_7a_term_years <= 0:
            errors.append("SBA 7(a) term must be positive")
        if self.working_capital_target < 0:
            errors.append("Working capital target cannot be negative")
        return errors


@dataclass
class EquipmentConfig:
    """Initial equipment and studio setup"""
    pottery_wheels: int = 8
    drying_racks: int = 10
    has_slab_roller: bool = True
    has_pug_mill: bool = False
    clay_traps: int = 3
    
    def validate(self) -> List[str]:
        errors = []
        if self.pottery_wheels <= 0:
            errors.append("Must have at least one pottery wheel")
        if self.drying_racks <= 0:
            errors.append("Must have at least one drying rack")
        if self.clay_traps <= 0:
            errors.append("Must have at least one clay trap")
        return errors


@dataclass
class MarketConfig:
    """Market response and pricing elasticity"""
    reference_price: float = 80.0
    join_price_elasticity: float = -1.4
    churn_price_elasticity: float = 1.2
    
    # Market pool sizes (monthly inflow potential)
    market_pools: Dict[str, int] = None
    
    def __post_init__(self):
        if self.market_pools is None:
            self.market_pools = {
                "community_studio": 4,
                "home_studio": 2, 
                "no_access": 3
            }
    
    def validate(self) -> List[str]:
        errors = []
        if self.reference_price <= 0:
            errors.append("Reference price must be positive")
        if self.join_price_elasticity > 0:
            errors.append("Join price elasticity should be negative (higher prices reduce demand)")
        if self.churn_price_elasticity < 0:
            errors.append("Churn price elasticity should be positive (higher prices increase churn)")
        return errors


@dataclass
class WorkshopConfig:
    """Workshop revenue stream configuration"""
    enabled: bool = True
    workshops_per_month: float = 4.0
    workshop_fee: float = 45.0
    avg_attendance: float = 12.0
    cost_per_event: float = 75.0
    conversion_rate: float = 0.15
    conversion_lag_months: int = 2
    
    def validate(self) -> List[str]:
        errors = []
        if self.workshops_per_month < 0:
            errors.append("Workshops per month cannot be negative")
        if self.workshop_fee < 0:
            errors.append("Workshop fee cannot be negative")
        if self.avg_attendance < 0:
            errors.append("Average attendance cannot be negative")
        if self.conversion_rate < 0 or self.conversion_rate > 1:
            errors.append("Conversion rate must be between 0 and 1")
        return errors


@dataclass
class ClassConfig:
    """Structured class revenue stream configuration"""
    enabled: bool = True
    calendar_mode: str = "monthly"  # "monthly" or "semester"
    cohorts_per_month: float = 2.0
    class_size_limit: int = 8
    class_price: float = 180.0
    conversion_rate: float = 0.25
    conversion_lag_months: int = 1
    
    def validate(self) -> List[str]:
        errors = []
        if self.calendar_mode not in ["monthly", "semester"]:
            errors.append("Calendar mode must be 'monthly' or 'semester'")
        if self.cohorts_per_month < 0:
            errors.append("Cohorts per month cannot be negative")
        if self.class_size_limit <= 0:
            errors.append("Class size limit must be positive")
        if self.class_price < 0:
            errors.append("Class price cannot be negative")
        if self.conversion_rate < 0 or self.conversion_rate > 1:
            errors.append("Conversion rate must be between 0 and 1")
        return errors


@dataclass
class EventConfig:
    """Special event revenue stream configuration"""
    enabled: bool = True
    ticket_price: float = 35.0
    max_events_per_month: int = 6
    base_events_per_month: float = 2.5
    attendee_range: List[int] = None
    staff_rate_per_hour: float = 15.0
    hours_per_event: float = 4.0
    material_cost_per_person: float = 8.0
    
    def __post_init__(self):
        if self.attendee_range is None:
            self.attendee_range = [8, 15]
    
    def validate(self) -> List[str]:
        errors = []
        if self.ticket_price < 0:
            errors.append("Ticket price cannot be negative")
        if self.max_events_per_month < 0:
            errors.append("Max events per month cannot be negative")
        if self.base_events_per_month < 0:
            errors.append("Base events per month cannot be negative")
        if len(self.attendee_range) != 2 or self.attendee_range[0] > self.attendee_range[1]:
            errors.append("Attendee range must be [min, max] with min <= max")
        return errors


@dataclass
class EconomicEnvironment:
    """External economic conditions affecting the studio"""
    downturn_prob_per_month: float = 0.08
    downturn_join_multiplier: float = 0.85
    downturn_churn_multiplier: float = 1.25
    
    def validate(self) -> List[str]:
        errors = []
        if self.downturn_prob_per_month < 0 or self.downturn_prob_per_month > 1:
            errors.append("Downturn probability must be between 0 and 1")
        if self.downturn_join_multiplier < 0:
            errors.append("Downturn join multiplier cannot be negative")
        if self.downturn_churn_multiplier < 0:
            errors.append("Downturn churn multiplier cannot be negative")
        return errors


@dataclass
class SimulationControls:
    """How the simulation runs"""
    num_simulations: int = 100
    time_horizon_months: int = 36
    random_seed: Optional[int] = 42
    
    def validate(self) -> List[str]:
        errors = []
        if self.num_simulations <= 0:
            errors.append("Number of simulations must be positive")
        if self.time_horizon_months <= 0:
            errors.append("Time horizon must be positive")
        return errors


class StudioConfig:
    """
    Consolidated configuration manager that groups all studio parameters
    and provides mapping to the existing simulator's expected format.
    """
    
    def __init__(self):
        self.business = BusinessFundamentals()
        self.financing = FinancingConfig()
        self.equipment = EquipmentConfig()
        self.market = MarketConfig()
        self.workshops = WorkshopConfig()
        self.classes = ClassConfig()
        self.events = EventConfig()
        self.economic = EconomicEnvironment()
        self.simulation = SimulationControls()
    
    def validate_all(self) -> Dict[str, List[str]]:
        """Validate all configuration sections and return any errors"""
        errors = {}
        
        sections = {
            "business": self.business,
            "financing": self.financing,
            "equipment": self.equipment,
            "market": self.market,
            "workshops": self.workshops,
            "classes": self.classes,
            "events": self.events,
            "economic": self.economic,
            "simulation": self.simulation
        }
        
        for section_name, section in sections.items():
            section_errors = section.validate()
            if section_errors:
                errors[section_name] = section_errors
        
        return errors
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """
        Convert the structured configuration to the flat parameter format
        expected by the existing simulator via final_batch_adapter.py
        """
        
        # Core simulation parameters
        legacy = {
            # Simulation controls
            "N_SIMULATIONS": self.simulation.num_simulations,
            "MONTHS": self.simulation.time_horizon_months,
            "RANDOM_SEED": self.simulation.random_seed,
            
            # Business fundamentals - convert to arrays as expected
            "RENT_SCENARIOS": np.array([self.business.monthly_rent], dtype=float),
            "OWNER_DRAW_SCENARIOS": [self.business.owner_monthly_compensation],
            "MAX_MEMBERS": self.business.max_members,
            "PRICE": self.business.membership_price,
            
            # Market dynamics
            "REFERENCE_PRICE": self.market.reference_price,
            "JOIN_PRICE_ELASTICITY": self.market.join_price_elasticity,
            "CHURN_PRICE_ELASTICITY": self.market.churn_price_elasticity,
            "MARKET_POOLS_INFLOW": self.market.market_pools,
            
            # Equipment (start-of-life)
            "N_WHEELS_START": self.equipment.pottery_wheels,
            "HAS_SLAB_ROLLER_START": self.equipment.has_slab_roller,
            "N_RACKS_START": self.equipment.drying_racks,
            "HAS_PUG_MILL_START": self.equipment.has_pug_mill,
            "N_CLAY_TRAPS_START": self.equipment.clay_traps,
            
            # Financing
            "RUNWAY_MONTHS": self.financing.runway_months,
            "LOAN_CONTINGENCY_PCT": self.financing.loan_contingency_pct,
            "EXTRA_BUFFER": self.financing.extra_buffer,
            "LOAN_504_ANNUAL_RATE": self.financing.loan_504_rate,
            "LOAN_504_TERM_YEARS": self.financing.loan_504_term_years,
            "LOAN_7A_ANNUAL_RATE": self.financing.loan_7a_rate,
            "LOAN_7A_TERM_YEARS": self.financing.loan_7a_term_years,
            "IO_MONTHS_504": self.financing.loan_504_io_months,
            "IO_MONTHS_7A": self.financing.loan_7a_io_months,
            "RESERVE_FLOOR": self.financing.reserve_floor,
            
            # Economic environment
            "DOWNTURN_PROB_PER_MONTH": self.economic.downturn_prob_per_month,
            "DOWNTURN_JOIN_MULT": self.economic.downturn_join_multiplier,
            "DOWNTURN_CHURN_MULT": self.economic.downturn_churn_multiplier,
            
            # Revenue streams - workshops
            "WORKSHOPS_ENABLED": self.workshops.enabled,
            "WORKSHOPS_PER_MONTH": self.workshops.workshops_per_month,
            "WORKSHOP_AVG_ATTENDANCE": self.workshops.avg_attendance,
            "WORKSHOP_FEE": self.workshops.workshop_fee,
            "WORKSHOP_COST_PER_EVENT": self.workshops.cost_per_event,
            "WORKSHOP_CONV_RATE": self.workshops.conversion_rate,
            "WORKSHOP_CONV_LAG_MO": self.workshops.conversion_lag_months,
            
            # Revenue streams - classes
            "CLASSES_ENABLED": self.classes.enabled,
            "CLASSES_CALENDAR_MODE": self.classes.calendar_mode,
            "CLASS_COHORTS_PER_MONTH": self.classes.cohorts_per_month,
            "CLASS_CAP_PER_COHORT": self.classes.class_size_limit,
            "CLASS_PRICE": self.classes.class_price,
            "CLASS_CONV_RATE": self.classes.conversion_rate,
            "CLASS_CONV_LAG_MO": self.classes.conversion_lag_months,
            
            # Revenue streams - events
            "EVENTS_ENABLED": self.events.enabled,
            "TICKET_PRICE": self.events.ticket_price,
            "EVENTS_MAX_PER_MONTH": self.events.max_events_per_month,
            "BASE_EVENTS_PER_MONTH_LAMBDA": self.events.base_events_per_month,
            "EVENT_STAFF_RATE_PER_HOUR": self.events.staff_rate_per_hour,
            "EVENT_HOURS_PER_EVENT": self.events.hours_per_event,
            "ATTENDEES_PER_EVENT_RANGE": self.events.attendee_range,
            "EVENT_CONSUMABLES_PER_PERSON": self.events.material_cost_per_person,
        }
        
        # Create scenario config for the simulator
        legacy["SCENARIO_CONFIGS"] = [{
            "name": "Beta_Test_Scenario",
            "capex_timing": "staged",  # Default to staged approach
            "grant_amount": 0.0,       # No grant by default
            "grant_month": None,
        }]
        
        return legacy
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        return {
            "business": asdict(self.business),
            "financing": asdict(self.financing),
            "equipment": asdict(self.equipment),
            "market": asdict(self.market),
            "workshops": asdict(self.workshops),
            "classes": asdict(self.classes),
            "events": asdict(self.events),
            "economic": asdict(self.economic),
            "simulation": asdict(self.simulation)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudioConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        if "business" in data:
            config.business = BusinessFundamentals(**data["business"])
        if "financing" in data:
            config.financing = FinancingConfig(**data["financing"])
        if "equipment" in data:
            config.equipment = EquipmentConfig(**data["equipment"])
        if "market" in data:
            config.market = MarketConfig(**data["market"])
        if "workshops" in data:
            config.workshops = WorkshopConfig(**data["workshops"])
        if "classes" in data:
            config.classes = ClassConfig(**data["classes"])
        if "events" in data:
            config.events = EventConfig(**data["events"])
        if "economic" in data:
            config.economic = EconomicEnvironment(**data["economic"])
        if "simulation" in data:
            config.simulation = SimulationControls(**data["simulation"])
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'StudioConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined configuration presets for different scenarios
SCENARIO_PRESETS = {
    "conservative": {
        "description": "Conservative assumptions for cautious planning",
        "modifications": {
            "economic": {"downturn_prob_per_month": 0.12},
            "market": {"join_price_elasticity": -1.8},
            "business": {"membership_price": 75.0},
        }
    },
    "optimistic": {
        "description": "Optimistic assumptions for best-case planning", 
        "modifications": {
            "economic": {"downturn_prob_per_month": 0.04},
            "market": {"join_price_elasticity": -1.0},
            "business": {"membership_price": 95.0},
        }
    },
    "realistic": {
        "description": "Balanced assumptions based on industry data",
        "modifications": {}  # Uses defaults
    }
}


def create_preset_config(preset_name: str) -> StudioConfig:
    """Create a configuration using one of the predefined presets"""
    if preset_name not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(SCENARIO_PRESETS.keys())}")
    
    config = StudioConfig()
    preset = SCENARIO_PRESETS[preset_name]
    
    # Apply modifications from preset
    for section_name, modifications in preset["modifications"].items():
        section = getattr(config, section_name)
        for param, value in modifications.items():
            setattr(section, param, value)
    
    return config
