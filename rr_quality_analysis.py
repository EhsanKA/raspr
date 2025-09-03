#!/usr/bin/env python3
"""
RR Data Quality Analysis and HR Comparison

This script analyzes the quality and consistency of RR interval data by comparing
it with the reported heart rate (HR) data. It investigates potential filtering,
noise, and preprocessing requirements.

Author: Generated for RASPR project
Date: September 2025
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_timestamp_to_pdt(timestamp):
    """Convert Unix timestamp to PDT timezone."""
    pdt_offset = timedelta(hours=-7)
    pdt_timezone = timezone(pdt_offset)
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt_utc.astimezone(pdt_timezone)

def analyze_rr_vs_hr_consistency(data):
    """
    Analyze consistency between RR intervals and reported HR.

    Args:
        data (list): List of data points with timestamps, HR, and RR intervals

    Returns:
        dict: Analysis results
    """
    print("Analyzing RR vs HR consistency...")

    # Extract data points that have both HR and RR
    valid_points = []
    for entry in data:
        if 'hr' in entry and entry['hr'] is not None and 'rr' in entry and entry['rr']:
            valid_points.append(entry)

    print(f"Found {len(valid_points)} data points with both HR and RR data")

    if len(valid_points) < 10:
        return {"error": "Insufficient data points for analysis"}

    # Calculate HR from RR intervals for each point
    hr_from_rr = []
    reported_hr = []
    timestamps = []
    rr_variability = []

    for entry in valid_points:
        rr_intervals = np.array(entry['rr'], dtype=float)
        if len(rr_intervals) > 0:
            # Calculate HR from RR (HR = 60 / RR in minutes)
            hr_calculated = 60000.0 / np.mean(rr_intervals)  # Convert ms to minutes
            hr_reported = entry['hr']

            hr_from_rr.append(hr_calculated)
            reported_hr.append(hr_reported)
            timestamps.append(entry['ts'])

            # Calculate RR variability (coefficient of variation)
            if len(rr_intervals) > 1:
                rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
                rr_variability.append(rr_cv)
            else:
                rr_variability.append(0)

    hr_from_rr = np.array(hr_from_rr)
    reported_hr = np.array(reported_hr)
    rr_variability = np.array(rr_variability)

    # Calculate differences and statistics
    hr_differences = hr_from_rr - reported_hr
    abs_differences = np.abs(hr_differences)

    analysis = {
        'hr_from_rr': hr_from_rr,
        'reported_hr': reported_hr,
        'hr_differences': hr_differences,
        'abs_differences': abs_differences,
        'timestamps': timestamps,
        'rr_variability': rr_variability,
        'stats': {
            'mean_difference': np.mean(hr_differences),
            'std_difference': np.std(hr_differences),
            'mean_abs_difference': np.mean(abs_differences),
            'max_abs_difference': np.max(abs_differences),
            'correlation': pearsonr(hr_from_rr, reported_hr)[0],
            'spearman_correlation': spearmanr(hr_from_rr, reported_hr)[0],
            'rmse': np.sqrt(np.mean(hr_differences**2))
        }
    }

    return analysis

def detect_hr_filtering(analysis):
    """
    Detect if HR data has been filtered or processed.

    Args:
        analysis (dict): Analysis results from analyze_rr_vs_hr_consistency

    Returns:
        dict: Filtering detection results
    """
    print("Detecting potential HR filtering...")

    hr_from_rr = analysis['hr_from_rr']
    reported_hr = analysis['reported_hr']
    differences = analysis['hr_differences']

    # Check for systematic bias
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # Check for smoothing/filtering indicators
    # 1. Reported HR should be smoother than calculated HR
    hr_from_rr_smoothness = np.std(np.diff(hr_from_rr))
    reported_hr_smoothness = np.std(np.diff(reported_hr))

    # 2. Check for outlier removal
    hr_from_rr_outliers = detect_outliers(hr_from_rr)
    reported_hr_outliers = detect_outliers(reported_hr)

    # 3. Check for rounding/quantization
    hr_from_rr_decimal = np.mean([len(str(x).split('.')[-1]) for x in hr_from_rr if '.' in str(x)])
    reported_hr_decimal = np.mean([len(str(x).split('.')[-1]) for x in reported_hr if '.' in str(x)])

    filtering_indicators = {
        'systematic_bias': {
            'mean_difference': mean_diff,
            'bias_direction': 'reported_higher' if mean_diff < 0 else 'calculated_higher',
            'bias_magnitude': abs(mean_diff)
        },
        'smoothness_comparison': {
            'hr_from_rr_smoothness': hr_from_rr_smoothness,
            'reported_hr_smoothness': reported_hr_smoothness,
            'smoothing_ratio': reported_hr_smoothness / hr_from_rr_smoothness if hr_from_rr_smoothness > 0 else float('inf')
        },
        'outlier_analysis': {
            'hr_from_rr_outliers': len(hr_from_rr_outliers),
            'reported_hr_outliers': len(reported_hr_outliers),
            'outlier_ratio': len(reported_hr_outliers) / len(hr_from_rr_outliers) if len(hr_from_rr_outliers) > 0 else 0
        },
        'precision_analysis': {
            'hr_from_rr_precision': hr_from_rr_decimal,
            'reported_hr_precision': reported_hr_decimal
        }
    }

    return filtering_indicators

def detect_outliers(data, threshold=3):
    """Detect outliers using z-score method."""
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return np.where(z_scores > threshold)[0]

def create_rr_quality_plots(analysis, filtering_indicators):
    """
    Create comprehensive plots for RR data quality analysis.

    Args:
        analysis (dict): Analysis results
        filtering_indicators (dict): Filtering detection results
    """
    print("Creating RR quality analysis plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RR Data Quality Analysis: HR vs RR Interval Consistency', fontsize=16, fontweight='bold')

    # Convert timestamps to PDT time strings for plotting
    time_strings = [convert_timestamp_to_pdt(ts).strftime('%H:%M:%S') for ts in analysis['timestamps']]
    time_indices = range(len(time_strings))

    # Plot 1: HR comparison over time
    ax1.plot(time_indices, analysis['hr_from_rr'], 'b-', label='HR from RR intervals', alpha=0.7, linewidth=1.5)
    ax1.plot(time_indices, analysis['reported_hr'], 'r-', label='Reported HR', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Heart Rate (BPM)')
    ax1.set_title('HR Comparison: Calculated vs Reported')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add some time labels
    if len(time_strings) > 10:
        step = len(time_strings) // 10
        ax1.set_xticks(range(0, len(time_strings), step))
        ax1.set_xticklabels([time_strings[i] for i in range(0, len(time_strings), step)], rotation=45)

    # Plot 2: HR differences over time
    ax2.plot(time_indices, analysis['hr_differences'], 'g-', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(time_indices, analysis['hr_differences'], 0,
                    where=(analysis['hr_differences'] >= 0), color='red', alpha=0.3, label='Calculated > Reported')
    ax2.fill_between(time_indices, analysis['hr_differences'], 0,
                    where=(analysis['hr_differences'] <= 0), color='blue', alpha=0.3, label='Reported > Calculated')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('HR Difference (BPM)')
    ax2.set_title('HR Differences: (Calculated - Reported)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter plot with correlation
    ax3.scatter(analysis['hr_from_rr'], analysis['reported_hr'], alpha=0.6, s=20)
    ax3.plot([min(analysis['hr_from_rr']), max(analysis['hr_from_rr'])],
             [min(analysis['hr_from_rr']), max(analysis['hr_from_rr'])],
             'r--', alpha=0.7, label='Perfect correlation')

    # Add correlation line
    z = np.polyfit(analysis['hr_from_rr'], analysis['reported_hr'], 1)
    p = np.poly1d(z)
    ax3.plot(analysis['hr_from_rr'], p(analysis['hr_from_rr']), 'g-', alpha=0.7, label='Linear fit')

    ax3.set_xlabel('HR from RR Intervals (BPM)')
    ax3.set_ylabel('Reported HR (BPM)')
    ax3.set_title('.3f')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: RR variability vs HR differences
    ax4.scatter(analysis['rr_variability'], analysis['abs_differences'], alpha=0.6, s=20)
    ax4.set_xlabel('RR Interval Variability (CV)')
    ax4.set_ylabel('Absolute HR Difference (BPM)')
    ax4.set_title('RR Variability vs HR Consistency')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rr_quality_analysis.png', dpi=300, bbox_inches='tight')
    print("RR quality analysis plot saved to rr_quality_analysis.png")
    plt.show()

def create_filtering_analysis_plot(filtering_indicators):
    """Create plot showing filtering analysis results."""
    print("Creating filtering analysis plot...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('HR Data Filtering Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Systematic bias
    bias_data = filtering_indicators['systematic_bias']
    ax1.bar(['Mean Difference'], [bias_data['mean_difference']], color='skyblue', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('HR Difference (BPM)')
    ax1.set_title(f'Systematic Bias\n({bias_data["bias_direction"]}: {bias_data["bias_magnitude"]:.2f} BPM)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Smoothness comparison
    smoothness_data = filtering_indicators['smoothness_comparison']
    methods = ['HR from RR', 'Reported HR']
    smoothness_values = [smoothness_data['hr_from_rr_smoothness'], smoothness_data['reported_hr_smoothness']]
    bars = ax2.bar(methods, smoothness_values, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Smoothness (Std of differences)')
    ax2.set_title('.2f')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Outlier analysis
    outlier_data = filtering_indicators['outlier_analysis']
    outlier_labels = ['HR from RR\nOutliers', 'Reported HR\nOutliers']
    outlier_values = [outlier_data['hr_from_rr_outliers'], outlier_data['reported_hr_outliers']]
    bars = ax3.bar(outlier_labels, outlier_values, color=['orange', 'green'], alpha=0.7)
    ax3.set_ylabel('Number of Outliers')
    ax3.set_ylabel('Number of Outliers')
    ax3.set_title(f'Outlier Analysis\nRatio: {outlier_data["outlier_ratio"]:.2f}')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, outlier_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{value}', ha='center', va='bottom')

    # Plot 4: Precision analysis
    precision_data = filtering_indicators['precision_analysis']
    precision_labels = ['HR from RR', 'Reported HR']
    precision_values = [precision_data['hr_from_rr_precision'], precision_data['reported_hr_precision']]
    bars = ax4.bar(precision_labels, precision_values, color=['purple', 'brown'], alpha=0.7)
    ax4.set_ylabel('Decimal Precision')
    ax4.set_title('Data Precision Analysis')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, precision_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05, '.1f', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('hr_filtering_analysis.png', dpi=300, bbox_inches='tight')
    print("HR filtering analysis plot saved to hr_filtering_analysis.png")
    plt.show()

def print_analysis_summary(analysis, filtering_indicators):
    """Print comprehensive analysis summary."""
    print("\n" + "="*80)
    print("RR DATA QUALITY ANALYSIS SUMMARY")
    print("="*80)

    stats = analysis['stats']

    print("\n1. BASIC STATISTICS:")
    print(f"   Data points analyzed: {len(analysis['hr_from_rr'])}")
    print(f"   Mean HR from RR: {np.mean(analysis['hr_from_rr']):.2f} BPM")
    print(f"   Mean reported HR: {np.mean(analysis['reported_hr']):.2f} BPM")
    print(f"   HR range from RR: {np.min(analysis['hr_from_rr']):.1f} - {np.max(analysis['hr_from_rr']):.1f} BPM")
    print(f"   HR range reported: {np.min(analysis['reported_hr']):.1f} - {np.max(analysis['reported_hr']):.1f} BPM")
    print(f"   RR variability (mean CV): {np.mean(analysis['rr_variability']):.4f}")

    print("\n2. CONSISTENCY ANALYSIS:")
    print(f"   Mean difference: {stats['mean_difference']:.2f} BPM")
    print(f"   Std of differences: {stats['std_difference']:.2f} BPM")
    print(f"   Mean absolute difference: {stats['mean_abs_difference']:.2f} BPM")
    print(f"   Max absolute difference: {stats['max_abs_difference']:.2f} BPM")
    print(f"   RMSE: {stats['rmse']:.2f} BPM")
    print(f"   Pearson correlation: {stats['correlation']:.3f}")
    print(f"   Spearman correlation: {stats['spearman_correlation']:.3f}")

    bias = filtering_indicators['systematic_bias']
    print("\n3. FILTERING INDICATORS:")
    print(f"   Systematic bias: {bias['mean_difference']:.2f} BPM")
    print(f"   Bias direction: {bias['bias_direction']}")
    print(f"   Bias magnitude: {bias['bias_magnitude']:.2f} BPM")

    smoothness = filtering_indicators['smoothness_comparison']
    print(f"   HR from RR smoothness: {smoothness['hr_from_rr_smoothness']:.4f}")
    print(f"   Reported HR smoothness: {smoothness['reported_hr_smoothness']:.4f}")
    print(f"   Smoothing ratio: {smoothness['smoothing_ratio']:.2f}")

    outliers = filtering_indicators['outlier_analysis']
    print(f"   Outliers - HR from RR: {outliers['hr_from_rr_outliers']}")
    print(f"   Outliers - Reported HR: {outliers['reported_hr_outliers']}")
    print(f"   Outlier ratio: {outliers['outlier_ratio']:.2f}")

    precision = filtering_indicators['precision_analysis']
    print(f"   HR from RR precision: {precision['hr_from_rr_precision']:.1f} decimal places")
    print(f"   Reported HR precision: {precision['reported_hr_precision']:.1f} decimal places")

    print("\n4. RECOMMENDATIONS:")
    if abs(stats['correlation']) > 0.8:
        print("   ✓ Good correlation between HR and RR data")
    else:
        print("   ⚠ Poor correlation - investigate data quality")

    if abs(bias['mean_difference']) < 2:
        print("   ✓ Small systematic bias")
    else:
        print("   ⚠ Significant systematic bias detected")

    if smoothness['smoothing_ratio'] < 0.8:
        print("   ✓ Reported HR appears smoother (possible filtering)")
    else:
        print("   ✓ Similar smoothness between HR sources")

    if outliers['outlier_ratio'] < 0.5:
        print("   ✓ Reported HR has fewer outliers (possible filtering)")
    else:
        print("   ✓ Similar outlier patterns")

def main():
    """Main function."""
    try:
        print("Loading heart rate data...")
        data = load_data('Work/H10_log_20250611_2133.json')

        # Perform RR vs HR consistency analysis
        analysis = analyze_rr_vs_hr_consistency(data)

        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return 1

        # Detect filtering characteristics
        filtering_indicators = detect_hr_filtering(analysis)

        # Create visualizations
        create_rr_quality_plots(analysis, filtering_indicators)
        create_filtering_analysis_plot(filtering_indicators)

        # Print comprehensive summary
        print_analysis_summary(analysis, filtering_indicators)

        print("\nAnalysis complete! Generated files:")
        print("  - rr_quality_analysis.png: Main quality analysis plots")
        print("  - hr_filtering_analysis.png: Filtering detection plots")

    except Exception as e:
        print(f"Error in RR quality analysis: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
