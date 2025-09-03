#!/usr/bin/env python3
"""
Data Visualization Script

This script creates connected scatter plots for:
1. RR intervals over time
2. Heart rate over time
3. Calculated breathing rate over time

The plots help visualize the relationships between heart rate variability,
heart rate, and the estimated breathing rate.

Author: Generated for RASPR project
Date: September 2025
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
import argparse
import os


def load_json_data(file_path):
    """
    Load JSON data from the specified file.
    
    Args:
        file_path (str): Path to the JSON file containing heart rate data
        
    Returns:
        list: List of dictionaries containing timestamp, heart rate, and RR intervals
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} data points from {file_path}")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")


def load_csv_data(file_path):
    """
    Load breathing rate data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing breathing rate data
        
    Returns:
        pandas.DataFrame: DataFrame with time and bpm columns
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} breathing rate measurements from {file_path}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def convert_timestamp_to_pdt(timestamp):
    """
    Convert Unix timestamp to PDT timezone.
    
    Args:
        timestamp (float): Unix timestamp
        
    Returns:
        datetime: DateTime object in PDT timezone
    """
    # PDT is UTC-7
    pdt_offset = timedelta(hours=-7)
    pdt_timezone = timezone(pdt_offset)
    
    # Convert from Unix timestamp to datetime
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    dt_pdt = dt_utc.astimezone(pdt_timezone)
    
    return dt_pdt


def process_json_data_for_plotting(data):
    """
    Process JSON data and prepare it for plotting.
    
    Args:
        data (list): List of data points from JSON
        
    Returns:
        tuple: (timestamps, heart_rates, rr_timestamps, rr_values)
    """
    timestamps = []
    heart_rates = []
    rr_timestamps = []
    rr_values = []
    
    for entry in data:
        ts = entry['ts']
        dt_pdt = convert_timestamp_to_pdt(ts)
        
        # Heart rate data
        if 'hr' in entry:
            timestamps.append(dt_pdt)
            heart_rates.append(entry['hr'])
        
        # RR interval data
        if 'rr' in entry and entry['rr']:
            for rr_value in entry['rr']:
                rr_timestamps.append(dt_pdt)
                rr_values.append(rr_value)
    
    return timestamps, heart_rates, rr_timestamps, rr_values


def process_csv_data_for_plotting(df, base_date=None):
    """
    Process CSV breathing rate data for plotting.
    
    Args:
        df (pandas.DataFrame): DataFrame with time and bpm columns
        base_date (datetime): Base date to combine with time strings
        
    Returns:
        tuple: (timestamps, breathing_rates)
    """
    if base_date is None:
        # Use today's date as base
        base_date = datetime.now().date()
    
    timestamps = []
    breathing_rates = []
    
    for _, row in df.iterrows():
        time_str = row['time']
        bpm = row['bpm']
        
        # Parse time string (HH:MM:SS)
        time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        
        # Combine with base date
        dt = datetime.combine(base_date, time_obj)
        
        # Convert to PDT timezone
        pdt_offset = timedelta(hours=-7)
        pdt_timezone = timezone(pdt_offset)
        dt_pdt = dt.replace(tzinfo=pdt_timezone)
        
        timestamps.append(dt_pdt)
        breathing_rates.append(bpm)
    
    return timestamps, breathing_rates


def create_plots(hr_data, rr_data, br_data, output_file='data_visualization.png'):
    """
    Create connected scatter plots for heart rate, RR intervals, and breathing rate.
    
    Args:
        hr_data (tuple): (timestamps, heart_rates)
        rr_data (tuple): (timestamps, rr_values)
        br_data (tuple): (timestamps, breathing_rates)
        output_file (str): Output file path for the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Heart Rate Variability and Breathing Rate Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Heart Rate
    hr_timestamps, heart_rates = hr_data
    if hr_timestamps and heart_rates:
        ax1.plot(hr_timestamps, heart_rates, 'o-', color='red', markersize=3, linewidth=1, alpha=0.7)
        ax1.set_ylabel('Heart Rate (BPM)', fontsize=12)
        ax1.set_title('Heart Rate Over Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(min(heart_rates) - 5, max(heart_rates) + 5)
    
    # Plot 2: RR Intervals
    rr_timestamps, rr_values = rr_data
    if rr_timestamps and rr_values:
        ax2.plot(rr_timestamps, rr_values, 'o-', color='blue', markersize=2, linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('RR Intervals (ms)', fontsize=12)
        ax2.set_title('RR Intervals (Inter-beat Intervals) Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(rr_values) - 20, max(rr_values) + 20)
    
    # Plot 3: Breathing Rate
    br_timestamps, breathing_rates = br_data
    if br_timestamps and breathing_rates:
        ax3.plot(br_timestamps, breathing_rates, 'o-', color='green', markersize=4, linewidth=1.5, alpha=0.8)
        ax3.set_ylabel('Breathing Rate (BPM)', fontsize=12)
        ax3.set_xlabel('Time (PDT)', fontsize=12)
        ax3.set_title('Calculated Breathing Rate Over Time', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(min(breathing_rates) - 2, max(breathing_rates) + 2)
    
    # Format x-axis for all subplots
    pdt_timezone = timezone(timedelta(hours=-7))
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=pdt_timezone))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show plot
    plt.show()


def create_correlation_plot(hr_data, rr_data, br_data, output_file='correlation_plot.png'):
    """
    Create correlation plots between different measurements.
    
    Args:
        hr_data (tuple): (timestamps, heart_rates)
        rr_data (tuple): (timestamps, rr_values)
        br_data (tuple): (timestamps, breathing_rates)
        output_file (str): Output file path for the correlation plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Data Correlations and Distributions', fontsize=16, fontweight='bold')
    
    # Extract data
    hr_timestamps, heart_rates = hr_data
    rr_timestamps, rr_values = rr_data
    br_timestamps, breathing_rates = br_data
    
    # Plot 1: Heart Rate Distribution
    if heart_rates:
        ax1.hist(heart_rates, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Heart Rate (BPM)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Heart Rate Distribution')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: RR Intervals Distribution
    if rr_values:
        ax2.hist(rr_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('RR Intervals (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('RR Intervals Distribution')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Breathing Rate Distribution
    if breathing_rates:
        ax3.hist(breathing_rates, bins=15, color='green', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Breathing Rate (BPM)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Breathing Rate Distribution')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: RR vs HR scatter (if we can align them)
    if heart_rates and rr_values and len(heart_rates) > 10:
        # Sample some points for scatter plot
        sample_size = min(len(heart_rates), 500)
        hr_sample = np.array(heart_rates[:sample_size])
        
        # Calculate mean RR for each HR measurement window
        rr_means = []
        for i in range(sample_size):
            # Get RR values around the same time
            start_idx = max(0, i * len(rr_values) // sample_size - 5)
            end_idx = min(len(rr_values), (i + 1) * len(rr_values) // sample_size + 5)
            if end_idx > start_idx:
                rr_window = rr_values[start_idx:end_idx]
                rr_means.append(np.mean(rr_window))
            else:
                rr_means.append(np.mean(rr_values))
        
        ax4.scatter(hr_sample, rr_means, alpha=0.6, color='purple', s=20)
        ax4.set_xlabel('Heart Rate (BPM)')
        ax4.set_ylabel('Mean RR Interval (ms)')
        ax4.set_title('Heart Rate vs RR Intervals')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Correlation plot saved to {output_file}")
    plt.show()


def main():
    """
    Main function to create visualization plots.
    """
    parser = argparse.ArgumentParser(description='Create visualization plots for heart rate and breathing rate data')
    parser.add_argument('--json', '-j',
                       default='Work/H10_log_20250611_2133.json',
                       help='Input JSON file path (default: Work/H10_log_20250611_2133.json)')
    parser.add_argument('--csv', '-c',
                       default='breathing_rate_output.csv',
                       help='Input CSV file path (default: breathing_rate_output.csv)')
    parser.add_argument('--output', '-o',
                       default='data_visualization.png',
                       help='Output plot file path (default: data_visualization.png)')
    parser.add_argument('--correlation', '-r',
                       default='correlation_plot.png',
                       help='Correlation plot file path (default: correlation_plot.png)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading data files...")
        json_data = load_json_data(args.json)
        csv_data = load_csv_data(args.csv)
        
        # Process data for plotting
        print("Processing data for visualization...")
        hr_timestamps, heart_rates, rr_timestamps, rr_values = process_json_data_for_plotting(json_data)
        
        # Use the first JSON timestamp to align CSV data
        if hr_timestamps:
            base_date = hr_timestamps[0].date()
            br_timestamps, breathing_rates = process_csv_data_for_plotting(csv_data, base_date)
        else:
            br_timestamps, breathing_rates = process_csv_data_for_plotting(csv_data)
        
        # Create plots
        print("Creating time series plots...")
        create_plots(
            (hr_timestamps, heart_rates),
            (rr_timestamps, rr_values),
            (br_timestamps, breathing_rates),
            args.output
        )
        
        print("Creating correlation plots...")
        create_correlation_plot(
            (hr_timestamps, heart_rates),
            (rr_timestamps, rr_values),
            (br_timestamps, breathing_rates),
            args.correlation
        )
        
        # Print summary statistics
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        if heart_rates:
            print(f"Heart Rate: {len(heart_rates)} measurements")
            print(f"  Range: {min(heart_rates):.1f} - {max(heart_rates):.1f} BPM")
            print(f"  Mean: {np.mean(heart_rates):.1f} BPM")
            print(f"  Std: {np.std(heart_rates):.1f} BPM")
        
        if rr_values:
            print(f"\nRR Intervals: {len(rr_values)} measurements")
            print(f"  Range: {min(rr_values):.1f} - {max(rr_values):.1f} ms")
            print(f"  Mean: {np.mean(rr_values):.1f} ms")
            print(f"  Std: {np.std(rr_values):.1f} ms")
        
        if breathing_rates:
            print(f"\nBreathing Rate: {len(breathing_rates)} measurements")
            print(f"  Range: {min(breathing_rates):.1f} - {max(breathing_rates):.1f} BPM")
            print(f"  Mean: {np.mean(breathing_rates):.1f} BPM")
            print(f"  Std: {np.std(breathing_rates):.1f} BPM")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
