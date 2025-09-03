#!/usr/bin/env python3
"""
Base Breathing Rate Estimation Module

This module provides the base interface and common utilities for all
breathing rate estimation methods.

Author: Modular breathing rate analysis
Date: September 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import welch


class BreathingRateEstimator(ABC):
    """
    Abstract base class for breathing rate estimation methods.

    This class defines the common interface that all breathing rate
    estimation methods must implement.
    """

    def __init__(self, name, description):
        """
        Initialize the breathing rate estimator.

        Args:
            name (str): Name of the estimation method
            description (str): Description of the method
        """
        self.name = name
        self.description = description

    @abstractmethod
    def estimate(self, rr_intervals, sampling_rate=4.0):
        """
        Estimate breathing rate from RR intervals.

        Args:
            rr_intervals (list): List of RR intervals in milliseconds
            sampling_rate (float): Sampling rate for interpolation (Hz)

        Returns:
            float: Estimated breathing rate in BPM, or np.nan if estimation fails
        """
        pass

    def validate_input(self, rr_intervals, min_intervals=5):
        """
        Validate input RR intervals.

        Args:
            rr_intervals (list): List of RR intervals
            min_intervals (int): Minimum number of intervals required

        Returns:
            bool: True if input is valid, False otherwise
        """
        if not rr_intervals or len(rr_intervals) < min_intervals:
            return False

        rr_array = np.array(rr_intervals, dtype=float)
        if np.any(rr_array <= 0) or np.any(np.isnan(rr_array)):
            return False

        return True

    def preprocess_rr_intervals(self, rr_intervals):
        """
        Preprocess RR intervals for analysis.

        Args:
            rr_intervals (list): Raw RR intervals in milliseconds

        Returns:
            tuple: (rr_seconds, time_points, instantaneous_hr)
        """
        rr_seconds = np.array(rr_intervals) / 1000.0  # Convert to seconds
        time_points = np.cumsum(rr_seconds)[:-1]  # Cumulative time points
        instantaneous_hr = 60.0 / rr_seconds  # Instantaneous heart rate

        return rr_seconds, time_points, instantaneous_hr

    def interpolate_signal(self, time_points, signal_values, sampling_rate=4.0):
        """
        Interpolate signal to uniform sampling rate.

        Args:
            time_points (np.array): Original time points
            signal_values (np.array): Signal values
            sampling_rate (float): Target sampling rate (Hz)

        Returns:
            tuple: (time_interp, signal_interp)
        """
        if len(time_points) < 2:
            return np.array([]), np.array([])

        time_interp = np.arange(time_points[0], time_points[-1], 1/sampling_rate)

        if len(time_interp) < 2:
            return np.array([]), np.array([])

        try:
            interp_func = interp1d(time_points, signal_values[:-1],
                                 kind='cubic', fill_value='extrapolate')
            signal_interp = interp_func(time_interp)
            return time_interp, signal_interp
        except:
            return np.array([]), np.array([])


class HRVFeatures:
    """
    Class for calculating heart rate variability features.
    """

    @staticmethod
    def calculate_rmssd(rr_intervals):
        """
        Calculate Root Mean Square of Successive Differences (RMSSD).

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            float: RMSSD value
        """
        if len(rr_intervals) < 2:
            return 0.0

        rr_array = np.array(rr_intervals, dtype=float)
        successive_diffs = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(successive_diffs**2))

        return rmssd

    @staticmethod
    def calculate_sdnn(rr_intervals):
        """
        Calculate Standard Deviation of NN intervals (SDNN).

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            float: SDNN value
        """
        if len(rr_intervals) < 2:
            return 0.0

        rr_array = np.array(rr_intervals, dtype=float)
        sdnn = np.std(rr_array, ddof=1)

        return sdnn

    @staticmethod
    def calculate_pnn50(rr_intervals):
        """
        Calculate Percentage of successive RR intervals that differ by >50ms (pNN50).

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            float: pNN50 percentage
        """
        if len(rr_intervals) < 2:
            return 0.0

        rr_array = np.array(rr_intervals, dtype=float)
        successive_diffs = np.abs(np.diff(rr_array))
        nn50_count = np.sum(successive_diffs > 50)
        pnn50 = (nn50_count / len(successive_diffs)) * 100

        return pnn50

    @staticmethod
    def calculate_cv(rr_intervals):
        """
        Calculate Coefficient of Variation of RR intervals.

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            float: Coefficient of variation as percentage
        """
        if len(rr_intervals) < 2:
            return 0.0

        rr_array = np.array(rr_intervals, dtype=float)
        mean_rr = np.mean(rr_array)
        std_rr = np.std(rr_array, ddof=1)

        if mean_rr > 0:
            cv = (std_rr / mean_rr) * 100
        else:
            cv = 0.0

        return cv

    @classmethod
    def extract_all_features(cls, rr_intervals):
        """
        Extract all HRV features from RR intervals.

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            dict: Dictionary containing all HRV features
        """
        return {
            'rmssd': cls.calculate_rmssd(rr_intervals),
            'sdnn': cls.calculate_sdnn(rr_intervals),
            'pnn50': cls.calculate_pnn50(rr_intervals),
            'cv_rr': cls.calculate_cv(rr_intervals),
            'mean_rr': np.mean(rr_intervals) if rr_intervals else 0.0,
            'count': len(rr_intervals)
        }


def convert_timestamp_to_pdt(timestamp):
    """
    Convert Unix timestamp to PDT timezone.

    Args:
        timestamp (float): Unix timestamp

    Returns:
        datetime: DateTime object in PDT timezone
    """
    pdt_offset = timedelta(hours=-7)
    pdt_timezone = timezone(pdt_offset)
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    dt_pdt = dt_utc.astimezone(pdt_timezone)

    return dt_pdt
