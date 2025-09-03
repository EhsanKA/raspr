#!/usr/bin/env python3
"""
Modular Breathing Rate Analysis

This script demonstrates the use of modular breathing rate estimation methods.
It provides detailed tracking and comparison of different approaches.

Author: Modular breathing rate analysis
Date: September 2025
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import argparse
import sys
import os

# Add the breathing_methods package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from breathing_methods import (
    get_estimator,
    list_available_methods,
    convert_timestamp_to_pdt
)


def load_data(json_file):
    """
    Load heart rate data from JSON file.

    Args:
        json_file (str): Path to JSON file

    Returns:
        list: List of data entries
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} data points from {json_file}")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []


def extract_rr_windows(data, window_seconds=30):
    """
    Extract RR interval windows from the data.

    Args:
        data (list): Raw data entries
        window_seconds (int): Window size in seconds

    Returns:
        list: List of (timestamp, rr_intervals) tuples
    """
    windows = []

    # Sort data by timestamp
    sorted_data = sorted(data, key=lambda x: x['ts'])

    start_time = sorted_data[0]['ts']
    current_time = start_time

    while current_time < sorted_data[-1]['ts']:
        end_time = current_time + window_seconds

        # Collect RR intervals in this window
        window_rr = []
        for entry in sorted_data:
            if current_time <= entry['ts'] < end_time:
                if 'rr' in entry and entry['rr']:
                    window_rr.extend(entry['rr'])

        if window_rr:
            dt_pdt = convert_timestamp_to_pdt(current_time)
            windows.append((dt_pdt, window_rr))

        current_time = end_time

    print(f"✅ Extracted {len(windows)} RR interval windows ({window_seconds}s each)")
    return windows


def analyze_window_detailed(window_data, method_name, window_idx):
    """
    Perform detailed analysis of a single window.

    Args:
        window_data (tuple): (timestamp, rr_intervals)
        method_name (str): Name of estimation method
        window_idx (int): Window index for tracking

    Returns:
        dict: Detailed analysis results
    """
    timestamp, rr_intervals = window_data

    # Get estimator
    estimator = get_estimator(method_name)
    if not estimator:
        return {"error": f"Unknown method: {method_name}"}

    # Basic RR statistics
    rr_stats = {
        "count": len(rr_intervals),
        "mean": np.mean(rr_intervals),
        "std": np.std(rr_intervals),
        "min": min(rr_intervals),
        "max": max(rr_intervals),
        "cv": np.std(rr_intervals) / np.mean(rr_intervals) * 100 if rr_intervals else 0
    }

    # Estimate breathing rate
    breathing_rate = estimator.estimate(rr_intervals)

    # Get method-specific details
    if hasattr(estimator, 'get_feature_importance') and method_name == 'hrv_time_domain':
        method_details = estimator.get_feature_importance(rr_intervals)
    elif hasattr(estimator, 'get_spectral_details') and method_name == 'spectral_analysis':
        method_details = estimator.get_spectral_details(rr_intervals)
    elif hasattr(estimator, 'get_statistical_details') and method_name == 'statistical_baseline':
        method_details = estimator.get_statistical_details(rr_intervals)
    else:
        method_details = {"method": estimator.name}

    return {
        "window_idx": window_idx,
        "timestamp": timestamp.strftime("%H:%M:%S"),
        "rr_stats": rr_stats,
        "breathing_rate": breathing_rate,
        "method_details": method_details,
        "method_name": estimator.name
    }


def analyze_all_methods(data, methods_to_test=None):
    """
    Analyze data using all available methods.

    Args:
        methods_to_test (list): List of method names to test, or None for all

    Returns:
        dict: Analysis results for all methods
    """
    if methods_to_test is None:
        methods_to_test = list_available_methods()

    print(f"🔬 Testing methods: {methods_to_test}")

    # Extract RR windows
    rr_windows = extract_rr_windows(data)

    if not rr_windows:
        print("❌ No RR windows extracted")
        return {}

    results = {}

    for method_name in methods_to_test:
        print(f"\n📊 Analyzing with {method_name}...")
        method_results = []

        for i, window in enumerate(rr_windows):
            if i >= 5:  # Limit to first 5 windows for detailed analysis
                break

            result = analyze_window_detailed(window, method_name, i)
            method_results.append(result)

            # Print summary for this window
            br = result['breathing_rate']
            rr_count = result['rr_stats']['count']
            rr_mean = result['rr_stats']['mean']
            print(f"  Window {i}: BR={br} BPM, RR_count={rr_count}, RR_mean={rr_mean:.1f}ms")

        results[method_name] = method_results

    return results


def print_detailed_comparison(results):
    """
    Print detailed comparison of all methods.

    Args:
        results (dict): Results from analyze_all_methods
    """
    print("\n" + "="*80)
    print("📈 DETAILED METHOD COMPARISON")
    print("="*80)

    for method_name, method_results in results.items():
        print(f"\n🔍 {method_name.upper()}")
        print("-" * 50)

        if not method_results:
            print("No results available")
            continue

        # Calculate statistics
        br_values = [r['breathing_rate'] for r in method_results if not np.isnan(r['breathing_rate'])]
        valid_count = len(br_values)

        if valid_count > 0:
            print(f"Valid measurements: {valid_count}/{len(method_results)}")
            print(f"Breathing rate range: {min(br_values):.1f} - {max(br_values):.1f} BPM")
            print(f"Mean breathing rate: {np.mean(br_values):.1f} BPM")
            print(f"Std breathing rate: {np.std(br_values):.1f} BPM")

            # Physiological assessment
            normal_range = [12, 20]
            normal_count = sum(1 for br in br_values if normal_range[0] <= br <= normal_range[1])
            normal_percentage = normal_count / valid_count * 100
            print(f"Physiological range (12-20 BPM): {normal_count}/{valid_count} ({normal_percentage:.1f}%)")

        # Show first window details
        if method_results:
            first_result = method_results[0]
            print("\nFirst window details:")
            print(f"  Timestamp: {first_result['timestamp']}")
            print(f"  RR count: {first_result['rr_stats']['count']}")
            print(f"  RR mean: {first_result['rr_stats']['mean']:.1f} ms")
            print(f"  RR CV: {first_result['rr_stats']['cv']:.1f}%")
            print(f"  Breathing rate: {first_result['breathing_rate']} BPM")


def demonstrate_method_details(results, method_name, window_idx=0):
    """
    Demonstrate detailed analysis for a specific method and window.

    Args:
        results (dict): Results from analyze_all_methods
        method_name (str): Method name to demonstrate
        window_idx (int): Window index to show details for
    """
    if method_name not in results:
        print(f"❌ Method {method_name} not found in results")
        return

    method_results = results[method_name]

    if window_idx >= len(method_results):
        print(f"❌ Window {window_idx} not available (max: {len(method_results)-1})")
        return

    result = method_results[window_idx]
    method_details = result['method_details']

    print(f"\n🔬 DETAILED ANALYSIS: {method_name.upper()} - Window {window_idx}")
    print("="*80)

    print(f"Timestamp: {result['timestamp']}")
    print(f"Breathing Rate: {result['breathing_rate']} BPM")
    print()

    print("RR Interval Statistics:")
    rr_stats = result['rr_stats']
    print(f"  Count: {rr_stats['count']}")
    print(f"  Mean: {rr_stats['mean']:.1f} ms")
    print(f"  Std: {rr_stats['std']:.1f} ms")
    print(f"  CV: {rr_stats['cv']:.1f}%")
    print(f"  Range: {rr_stats['min']:.1f} - {rr_stats['max']:.1f} ms")
    print()

    # Method-specific details
    if 'hrv_features' in method_details:
        print("HRV Features:")
        hrv = method_details['hrv_features']
        print(f"  RMSSD: {hrv['rmssd']:.2f} ms")
        print(f"  SDNN: {hrv['sdnn']:.2f} ms")
        print(f"  pNN50: {hrv['pnn50']:.2f}%")
        print(f"  CV_RR: {hrv['cv_rr']:.2f}%")

        if 'factors' in method_details:
            print("\nCalculation Factors:")
            factors = method_details['factors']
            for factor_name, factor_value in factors.items():
                print(f"  {factor_name}: {factor_value:.3f}")

    elif 'frequency_bands' in method_details:
        print("Spectral Analysis:")
        for band_name, band_info in method_details['frequency_bands'].items():
            print(f"  {band_name}: {band_info['peak_frequency']:.3f} Hz ({band_info['equivalent_bpm']:.1f} BPM)")

    elif 'method_contributions' in method_details:
        print("Statistical Method Contributions:")
        contributions = method_details['method_contributions']
        for method_type, method_info in contributions.items():
            print(f"  {method_type}: {method_info.get('breathing_rate', 'N/A')} BPM")

    if 'selection_logic' in method_details:
        print(f"\nSelection Logic: {method_details['selection_logic']}")

    if 'reliability_assessment' in method_details:
        print(f"Reliability: {method_details['reliability_assessment']}")


def main():
    """
    Main function for modular breathing rate analysis.
    """
    parser = argparse.ArgumentParser(description='Modular breathing rate analysis')
    parser.add_argument('--input', '-i',
                       default='Work/H10_log_20250611_2133.json',
                       help='Input JSON file path')
    parser.add_argument('--methods', '-m',
                       nargs='+',
                       choices=list_available_methods(),
                       help='Methods to test (default: all)')
    parser.add_argument('--detailed', '-d',
                       action='store_true',
                       help='Show detailed analysis for first method')
    parser.add_argument('--window', '-w',
                       type=int, default=0,
                       help='Window index for detailed analysis (default: 0)')

    args = parser.parse_args()

    print("🫁 MODULAR BREATHING RATE ANALYSIS")
    print("="*80)

    # Load data
    data = load_data(args.input)
    if not data:
        return 1

    # Analyze with specified methods
    methods_to_test = args.methods if args.methods else None
    results = analyze_all_methods(data, methods_to_test)

    if not results:
        print("❌ No analysis results generated")
        return 1

    # Print comparison
    print_detailed_comparison(results)

    # Show detailed analysis if requested
    if args.detailed and results:
        first_method = list(results.keys())[0]
        demonstrate_method_details(results, first_method, args.window)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("💡 Use --detailed flag to see method-specific calculations")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
