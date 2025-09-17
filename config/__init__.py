#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:37:13 2025

@author: harshadghodke
"""

# config/__init__.py
"""Configuration module for pottery studio simulator."""

from .parameters import CONSOLIDATED_PARAM_SPECS, CONSOLIDATED_GROUPS, consolidate_build_overrides
from .scenarios import SCENARIOS, STRATEGIES, DEFAULT_CAPEX_ITEMS

__all__ = [
    'CONSOLIDATED_PARAM_SPECS',
    'CONSOLIDATED_GROUPS', 
    'consolidate_build_overrides',
    'SCENARIOS',
    'STRATEGIES',
    'DEFAULT_CAPEX_ITEMS'
]