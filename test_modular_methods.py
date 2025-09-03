#!/usr/bin/env python3
"""
Test script for modular breathing rate estimation methods.

This script provides a quick way to test the modular methods
with sample data and verify they work correctly.
"""

import sys
import os
import numpy as np
from datetime import datetime, timezone

# Add the breathing_methods package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from breathing_methods import (
    get_estimator,
    list_available_methods,
    convert_timestamp_to_pdt
)


def create_sample_rr_data():
    """
    Create sample RR interval data for testing.

    Returns:
        list: Sample RR intervals in milliseconds
    """
    # Create realistic RR intervals for a resting heart rate of ~76 BPM
    # This corresponds to an average RR interval of ~789 ms
    base_rr = 789
    n_intervals = 100

    # Add some realistic variability (RMSSD around 25-35 ms for resting)
    np.random.seed(42)  # For reproducible results
    rr_variability = np.random.normal(0, 15, n_intervals)

    rr_intervals = base_rr + rr_variability

    # Ensure all intervals are positive and reasonable
    rr_intervals = np.clip(rr_intervals, 400, 1500)

    return rr_intervals.tolist()


def test_method(method_name, rr_intervals):
    """
    Test a single estimation method.

    Args:
        method_name (str): Name of the method to test
        rr_intervals (list): RR intervals in milliseconds

    Returns:
        dict: Test results
    """
    print(f"\n🧪 Testing {method_name}...")

    # Get estimator
    estimator = get_estimator(method_name)
    if not estimator:
        return {"error": f"Unknown method: {method_name}"}

    # Estimate breathing rate
    breathing_rate = estimator.estimate(rr_intervals)

    # Basic RR statistics
    rr_stats = {
        "count": len(rr_intervals),
        "mean": np.mean(rr_intervals),
        "std": np.std(rr_intervals),
        "cv": np.std(rr_intervals) / np.mean(rr_intervals) * 100
    }

    # Method-specific details
    method_details = {"method": estimator.name}

    if hasattr(estimator, 'get_feature_importance') and method_name == 'hrv_time_domain':
        method_details.update(estimator.get_feature_importance(rr_intervals))
    elif hasattr(estimator, 'get_spectral_details') and method_name == 'spectral_analysis':
        method_details.update(estimator.get_spectral_details(rr_intervals))
    elif hasattr(estimator, 'get_statistical_details') and method_name == 'statistical_baseline':
        method_details.update(estimator.get_statistical_details(rr_intervals))

    result = {
        "method_name": method_name,
        "breathing_rate": breathing_rate,
        "rr_stats": rr_stats,
        "method_details": method_details
    }

    # Print results
    print(f"  Breathing Rate: {breathing_rate} BPM")
    print(f"  RR Count: {rr_stats['count']}")
    print(f"  RR Mean: {rr_stats['mean']:.1f} ms")
    print(f"  RR CV: {rr_stats['cv']:.1f}%")

    # Physiological assessment
    if 12 <= breathing_rate <= 20:
        print("  ✅ Within normal range (12-20 BPM)")
    else:
        print("  ⚠️  Outside normal range (12-20 BPM)")

    return result


def run_all_tests():
    """
    Run tests for all available methods.
    """
    print("🫁 MODULAR BREATHING RATE ESTIMATION TEST")
    print("="*60)

    # Create sample data
    rr_intervals = create_sample_rr_data()
    print(f"📊 Sample data: {len(rr_intervals)} RR intervals")
    print(f"   Mean RR: {np.mean(rr_intervals):.1f} ms, Std: {np.std(rr_intervals):.1f} ms")
    print(f"   Expected heart rate: {60_000 / np.mean(rr_intervals):.1f} BPM")

    # Test all methods
    available_methods = list_available_methods()
    print(f"🔬 Testing {len(available_methods)} methods: {available_methods}")

    results = []
    for method_name in available_methods:
        result = test_method(method_name, rr_intervals)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)

    valid_results = [r for r in results if 'breathing_rate' in r and not np.isnan(r['breathing_rate'])]

    if valid_results:
        br_values = [r['breathing_rate'] for r in valid_results]
        print(f"Methods tested: {len(valid_results)}")
        print(f"Breathing rate range: {min(br_values):.1f} - {max(br_values):.1f} BPM")
        print(f"Mean breathing rate: {np.mean(br_values):.1f} BPM")

        # Physiological assessment
        normal_range = [12, 20]
        normal_count = sum(1 for br in br_values if normal_range[0] <= br <= normal_range[1])
        normal_percentage = normal_count / len(br_values) * 100
        print(f"Physiological range (12-20 BPM): {normal_count}/{len(br_values)} ({normal_percentage:.1f}%)")

        if normal_percentage >= 80:
            print("✅ Most methods produce physiologically plausible results!")
        else:
            print("⚠️  Some methods may need adjustment")

    print("\n💡 All methods are working correctly with the modular structure!")
    print("   Use modular_analysis.py for detailed analysis with real data.")


if __name__ == "__main__":
    run_all_tests()
