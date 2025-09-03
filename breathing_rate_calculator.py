#!/usr/bin/env python3
"""
Breathing Rate Calculator

This script calculates breathing rate at 15-second intervals from heart rate variability data.
The breathing rate is derived from inter-beat intervals (RR intervals) using respiratory 
sinus arrhythmia analysis.

Author: Generated for RASPR project
Date: September 2025
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from scipy import signal
from sklearn.preprocessing import StandardScaler
import csv
import argparse
import os


def load_data(file_path):
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


def extract_rr_features(rr_intervals):
    """
    Extract heart rate variability features from RR intervals.
    
    Args:
        rr_intervals (list): List of RR intervals in milliseconds
        
    Returns:
        dict: Dictionary containing HRV features
    """
    if not rr_intervals or len(rr_intervals) < 2:
        return {
            'rmssd': 0.0,
            'sdnn': 0.0,
            'pnn50': 0.0,
            'mean_rr': 0.0,
            'cv_rr': 0.0
        }
    
    rr_array = np.array(rr_intervals, dtype=float)
    
    # Basic time-domain features
    mean_rr = np.mean(rr_array)
    sdnn = np.std(rr_array, ddof=1)
    
    # RMSSD - Root mean square of successive differences
    if len(rr_array) > 1:
        successive_diffs = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(successive_diffs**2))
    else:
        rmssd = 0.0
    
    # pNN50 - Percentage of successive RR intervals that differ by more than 50ms
    if len(rr_array) > 1:
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100
    else:
        pnn50 = 0.0
    
    # Coefficient of variation
    cv_rr = (sdnn / mean_rr * 100) if mean_rr > 0 else 0.0
    
    return {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'mean_rr': mean_rr,
        'cv_rr': cv_rr
    }


def estimate_breathing_rate_from_hrv(hrv_features, baseline_br=16.0):
    """
    Estimate breathing rate from heart rate variability features.
    
    This function uses a heuristic approach based on the relationship between
    HRV and respiratory patterns. The breathing rate is estimated using:
    1. RMSSD (respiratory sinus arrhythmia indicator)
    2. Coefficient of variation of RR intervals
    3. pNN50 (autonomic activity indicator)
    
    Args:
        hrv_features (dict): Dictionary of HRV features
        baseline_br (float): Baseline breathing rate (breaths per minute)
        
    Returns:
        float: Estimated breathing rate in breaths per minute
    """
    rmssd = hrv_features['rmssd']
    cv_rr = hrv_features['cv_rr']
    pnn50 = hrv_features['pnn50']
    
    # For resting conditions, RMSSD typically ranges from 10-50 ms
    # Higher RMSSD generally indicates better vagal tone and respiratory modulation
    # The relationship is more nuanced than the previous linear model
    
    # Base adjustment based on RMSSD (corrected relationship)
    if rmssd < 10:
        # Low RMSSD - poor respiratory modulation, slightly faster breathing
        rmssd_factor = 1.1
    elif rmssd < 20:
        # Normal RMSSD range - good respiratory modulation
        rmssd_factor = 1.0
    elif rmssd < 40:
        # High RMSSD - strong respiratory modulation, slightly slower breathing
        rmssd_factor = 0.9
    else:
        # Very high RMSSD - very strong modulation, slower breathing
        rmssd_factor = 0.8
    
    # Adjust based on coefficient of variation (respiratory regularity)
    if cv_rr > 0:
        # Lower CV indicates more regular breathing (closer to baseline)
        cv_factor = max(0.8, min(1.2, 1.0 - (cv_rr - 3.0) / 10.0))
    else:
        cv_factor = 1.0
    
    # Adjust based on pNN50 (parasympathetic activity)
    if pnn50 > 0:
        pnn50_factor = max(0.9, min(1.1, 1.0 + (pnn50 - 5.0) / 20.0))
    else:
        pnn50_factor = 1.0
    
    # Combine factors
    breathing_rate = baseline_br * rmssd_factor * cv_factor * pnn50_factor
    
    # Constrain to physiological range (8-25 breaths per minute for resting)
    breathing_rate = max(8.0, min(25.0, breathing_rate))
    
    return breathing_rate


def process_data_in_intervals(data, interval_seconds=30):
    """
    Process data in specified time intervals and calculate breathing rate.
    
    Args:
        data (list): List of data points with timestamps and RR intervals
        interval_seconds (int): Time interval in seconds for aggregation
        
    Returns:
        list: List of tuples (time_str, breathing_rate)
    """
    if not data:
        return []
    
    # Sort data by timestamp
    sorted_data = sorted(data, key=lambda x: x['ts'])
    
    # Get start and end times
    start_time = sorted_data[0]['ts']
    end_time = sorted_data[-1]['ts']
    
    results = []
    current_time = start_time
    
    while current_time < end_time:
        interval_end = current_time + interval_seconds
        
        # Collect all RR intervals in this time window
        interval_rr = []
        interval_data_points = []
        
        for entry in sorted_data:
            if current_time <= entry['ts'] < interval_end:
                if 'rr' in entry and entry['rr']:
                    interval_rr.extend(entry['rr'])
                    interval_data_points.append(entry)
        
        if interval_rr:
            # Extract HRV features from collected RR intervals
            hrv_features = extract_rr_features(interval_rr)
            
            # Estimate breathing rate
            breathing_rate = estimate_breathing_rate_from_hrv(hrv_features)
            
            # Convert timestamp to PDT and format as HH:MM:SS
            dt_pdt = convert_timestamp_to_pdt(current_time)
            time_str = dt_pdt.strftime('%H:%M:%S')
            
            results.append((time_str, round(breathing_rate, 1)))
        
        current_time = interval_end
    
    return results


def save_results_to_csv(results, output_file='breathing_rate_output.csv'):
    """
    Save results to a CSV file.
    
    Args:
        results (list): List of tuples (time_str, breathing_rate)
        output_file (str): Path to output CSV file
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'bpm'])  # Header
        
        for time_str, breathing_rate in results:
            writer.writerow([time_str, breathing_rate])
    
    print(f"Results saved to {output_file}")
    print(f"Generated {len(results)} breathing rate measurements")


def main():
    """
    Main function to orchestrate the breathing rate calculation.
    """
    parser = argparse.ArgumentParser(description='Calculate breathing rate from heart rate variability data')
    parser.add_argument('--input', '-i', 
                       default='Work/H10_log_20250611_2133.json',
                       help='Input JSON file path (default: Work/H10_log_20250611_2133.json)')
    parser.add_argument('--output', '-o',
                       default='breathing_rate_output.csv',
                       help='Output CSV file path (default: breathing_rate_output.csv)')
    parser.add_argument('--interval', '-t',
                       type=int, default=30,
                       help='Time interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print("Loading heart rate data...")
        data = load_data(args.input)
        
        # Process data and calculate breathing rates
        print(f"Processing data in {args.interval}-second intervals...")
        results = process_data_in_intervals(data, args.interval)
        
        # Save results
        save_results_to_csv(results, args.output)
        
        # Display sample results
        print("\nSample results:")
        print("Time\t\tBreaths/min")
        print("-" * 25)
        for i, (time_str, br) in enumerate(results[:10]):
            print(f"{time_str}\t{br}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more entries")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
