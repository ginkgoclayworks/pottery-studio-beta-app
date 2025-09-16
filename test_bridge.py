#!/usr/bin/env python3
"""
Simple test to make sure our bridge to the existing simulator works
"""

import sys
import os

# Add the parent directory so we can find the existing simulator files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config_manager import StudioConfig
from core_simulator import run_simulation

def test_basic_simulation():
    """Run a very simple test simulation"""
    print("Creating basic studio configuration...")
    
    # Create a configuration with default values
    config = StudioConfig()
    
    # Make it run faster for testing
    config.simulation.num_simulations = 5  # Just 5 runs instead of 100
    config.simulation.time_horizon_months = 12  # Just 1 year instead of 3
    
    # Show what we're testing with
    print(f"Testing with:")
    print(f"  Monthly rent: ${config.business.monthly_rent:,}")
    print(f"  Membership price: ${config.business.membership_price}")
    print(f"  Max members: {config.business.max_members}")
    print(f"  Simulations: {config.simulation.num_simulations}")
    print(f"  Time horizon: {config.simulation.time_horizon_months} months")
    
    print("\nRunning simulation...")
    
    try:
        # This is where we test if our bridge works
        results = run_simulation(config)
        
        # If we get here, it worked!
        print("SUCCESS! The bridge is working.")
        
        # Show some simple results
        summary = results.summary
        print(f"\nResults:")
        print(f"  Survival rate: {summary.survival_probability:.1%}")
        print(f"  Final cash: ${summary.median_final_cash:,.0f}")
        print(f"  Monthly revenue: ${summary.median_monthly_revenue:,.0f}")
        print(f"  Break-even month: {summary.break_even_month:.1f}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        print("\nThis means we need to fix something before building the beta app.")
        return False

if __name__ == "__main__":
    print("Testing the bridge between new interface and existing simulator...")
    print("=" * 60)
    
    success = test_basic_simulation()
    
    if success:
        print("\n" + "=" * 60)
        print("BRIDGE TEST PASSED!")
        print("We're ready to build the simplified beta app.")
    else:
        print("\n" + "=" * 60)
        print("BRIDGE TEST FAILED!")
        print("We need to fix the connection first.")