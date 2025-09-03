#!/usr/bin/env python3
"""
Breathing Methods Package

This package provides modular implementations of different breathing rate
estimation methods for heart rate variability analysis.

Available Methods:
- HRV Time-Domain: Uses heart rate variability features
- Spectral Analysis: Uses frequency domain analysis
- Statistical Baseline: Uses empirical statistical relationships

Author: Modular breathing rate analysis
Date: September 2025
"""

from .base_estimator import BreathingRateEstimator, HRVFeatures, convert_timestamp_to_pdt
from .hrv_time_domain import HRVTimeDomainEstimator
from .spectral_analysis import SpectralEstimator
from .statistical_baseline import StatisticalEstimator

__all__ = [
    'BreathingRateEstimator',
    'HRVFeatures',
    'HRVTimeDomainEstimator',
    'SpectralEstimator',
    'StatisticalEstimator',
    'convert_timestamp_to_pdt'
]

# Method registry for easy access
METHOD_REGISTRY = {
    'hrv_time_domain': HRVTimeDomainEstimator,
    'spectral_analysis': SpectralEstimator,
    'statistical_baseline': StatisticalEstimator
}

def get_estimator(method_name):
    """
    Get an estimator instance by name.

    Args:
        method_name (str): Name of the estimation method

    Returns:
        BreathingRateEstimator: Estimator instance, or None if not found
    """
    if method_name in METHOD_REGISTRY:
        return METHOD_REGISTRY[method_name]()
    else:
        available_methods = list(METHOD_REGISTRY.keys())
        print(f"Available methods: {available_methods}")
        return None

def list_available_methods():
    """
    List all available estimation methods.

    Returns:
        list: List of available method names
    """
    return list(METHOD_REGISTRY.keys())
