#!/usr/bin/env python3
"""
Alternative Breathing Rate Calculator

This script calculates breathing rate using frequency domain analysis of heart rate variability.
It uses spectral analysis (Welch's method) to identify respiratory frequency components.

Author: Alternative implementation for comparison
Date: September 2025
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
from datetime import datetime, timedelta
import pytz

def estimate_bpm(rr_list, min_ibis=5, fs=4.0, resp_low=0.1, resp_high=0.5):
    """
    Estimate breathing rate from RR intervals using frequency domain analysis.
    
    Args:
        rr_list (list): List of RR intervals in milliseconds
        min_ibis (int): Minimum number of intervals required
        fs (float): Sampling frequency for interpolation
        resp_low (float): Low frequency bound for respiratory band (Hz)
        resp_high (float): High frequency bound for respiratory band (Hz)
    
    Returns:
        float: Estimated breathing rate in BPM, or NaN if calculation fails
    """
    if len(rr_list) < min_ibis:
        return np.nan
    
    rr = np.array(rr_list) / 1000.0  # Convert ms to seconds
    if np.any(rr <= 0):
        return np.nan
    
    t = np.cumsum(rr)[:-1]  # Cumulative times for intervals (exclude last for alignment)
    ihr = 60.0 / rr  # Instantaneous heart rate
    
    if len(t) < min_ibis:
        return np.nan
    
    t_new = np.arange(t[0], t[-1], 1/fs)
    if len(t_new) < 2 * fs:  # Too short
        return np.nan
    
    interp_func = interp1d(t, ihr[:-1], kind='cubic', fill_value='extrapolate')
    ihr_interp = interp_func(t_new)
    
    f, pxx = welch(ihr_interp, fs=fs, nperseg=min(len(ihr_interp), 256))
    idx = (f >= resp_low) & (f <= resp_high)
    if not np.any(idx):
        return np.nan
    
    f_resp = f[idx]
    pxx_resp = pxx[idx]
    
    # Prioritize peaks in the physiological respiratory range (0.2-0.4 Hz = 12-24 BPM)
    # This is more likely to be true respiratory modulation than very low frequency peaks
    physio_idx = (f_resp >= 0.2) & (f_resp <= 0.4)
    if np.any(physio_idx):
        # Use the peak in the physiological range
        physio_peaks = pxx_resp[physio_idx]
        physio_freqs = f_resp[physio_idx]
        peak_idx = np.argmax(physio_peaks)
        f_resp_peak = physio_freqs[peak_idx]
    else:
        # Fallback to overall maximum if no physiological peaks found
        peak_idx = np.argmax(pxx_resp)
        f_resp_peak = f_resp[peak_idx]
    
    return round(f_resp_peak * 60, 1)

def main(input_file, output_file):
    """
    Main function to process data and calculate breathing rates.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output CSV file
    """
    # Load data (assume JSON list of dicts)
    with open(input_file, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # If CSV instead: df = pd.read_csv(input_file, converters={'rr': lambda x: eval(x) if x else []})
    
    df['ts'] = pd.to_datetime(df['ts'], unit='s', utc=True).dt.tz_convert('US/Pacific')  # PDT
    df = df.sort_values('ts').reset_index(drop=True)
    
    start_time = df['ts'].min().floor('S')
    end_time = df['ts'].max().ceil('S')
    windows = pd.date_range(start_time, end_time, freq='30S')
    
    results = []
    for i in range(len(windows) - 1):
        win_start = windows[i]
        win_end = windows[i + 1]
        win_df = df[(df['ts'] >= win_start) & (df['ts'] < win_end)]
        all_rr = [ibi for rr in win_df['rr'] if isinstance(rr, list) for ibi in rr]
        bpm = estimate_bpm(all_rr)
        time_str = win_start.strftime('%H:%M:%S')
        results.append({'time': time_str, 'bpm': bpm if not np.isnan(bpm) else 0.0})
    
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python alternative_breathing_calculator.py <input_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
