#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:40:18 2025

@author: harshadghodke
"""


# analysis/__init__.py
"""Analysis and metrics module."""

from .metrics import compute_kpis_from_cell, summarize_cell, calculate_business_viability_score

__all__ = [
    'compute_kpis_from_cell',
    'summarize_cell',
    'calculate_business_viability_score'
]
