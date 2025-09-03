#!/usr/bin/env python3
"""
Spectral Analysis Breathing Rate Estimator

This module implements breathing rate estimation using frequency domain analysis
of heart rate variability. It uses Welch's method to identify respiratory frequency
components in the power spectral density of RR intervals.

Author: Modular breathing rate analysis
Date: September 2025
"""

import numpy as np
from scipy.signal import welch
from .base_estimator import BreathingRateEstimator


class SpectralEstimator(BreathingRateEstimator):
    """
    Breathing rate estimation using spectral analysis.

    This method analyzes the frequency content of heart rate variability to
    identify respiratory modulation. It prioritizes peaks in the physiological
    respiratory frequency range (0.2-0.4 Hz = 12-24 BPM).
    """

    def __init__(self):
        super().__init__(
            name="Spectral Analysis",
            description="Estimates breathing rate using frequency domain analysis of HRV"
        )

    def estimate(self, rr_intervals, sampling_rate=4.0):
        """
        Estimate breathing rate using spectral analysis.

        Args:
            rr_intervals (list): List of RR intervals in milliseconds
            sampling_rate (float): Sampling rate for interpolation (Hz)

        Returns:
            float: Estimated breathing rate in BPM, or np.nan if estimation fails
        """
        # Validate input
        if not self.validate_input(rr_intervals):
            return np.nan

        # Preprocess RR intervals
        rr_seconds, time_points, instantaneous_hr = self.preprocess_rr_intervals(rr_intervals)

        # Interpolate signal
        time_interp, hr_interp = self.interpolate_signal(time_points, instantaneous_hr, sampling_rate)

        if len(time_interp) < 2:
            return np.nan

        # Perform spectral analysis
        breathing_rate = self._spectral_analysis(hr_interp, sampling_rate)

        return breathing_rate

    def _spectral_analysis(self, hr_signal, sampling_rate):
        """
        Perform spectral analysis to find respiratory frequency.

        Args:
            hr_signal (np.array): Interpolated heart rate signal
            sampling_rate (float): Sampling rate (Hz)

        Returns:
            float: Estimated breathing rate in BPM
        """
        # Define frequency bands
        resp_low = 0.1   # 6 BPM (very low frequency)
        resp_high = 0.5  # 30 BPM (high frequency)

        # Compute power spectral density using Welch's method
        try:
            f, pxx = welch(hr_signal, fs=sampling_rate,
                          nperseg=min(len(hr_signal), 256),
                          scaling='density')
        except:
            return np.nan

        # Find respiratory frequency band
        resp_idx = (f >= resp_low) & (f <= resp_high)

        if not np.any(resp_idx):
            return np.nan

        f_resp = f[resp_idx]
        pxx_resp = pxx[resp_idx]

        # Prioritize peaks in physiological respiratory range (0.2-0.4 Hz = 12-24 BPM)
        physio_idx = (f_resp >= 0.2) & (f_resp <= 0.4)

        if np.any(physio_idx):
            # Use peak in physiological range
            physio_peaks = pxx_resp[physio_idx]
            physio_freqs = f_resp[physio_idx]
            peak_idx = np.argmax(physio_peaks)
            respiratory_freq = physio_freqs[peak_idx]
        else:
            # Fallback to overall maximum in respiratory band
            peak_idx = np.argmax(pxx_resp)
            respiratory_freq = f_resp[peak_idx]

        # Convert frequency to breathing rate (BPM)
        breathing_rate = respiratory_freq * 60

        return round(breathing_rate, 1)

    def get_spectral_details(self, rr_intervals, sampling_rate=4.0):
        """
        Get detailed spectral analysis for debugging and visualization.

        Args:
            rr_intervals (list): RR intervals in milliseconds
            sampling_rate (float): Sampling rate for interpolation (Hz)

        Returns:
            dict: Detailed spectral analysis results
        """
        if not self.validate_input(rr_intervals):
            return {"error": "Invalid input"}

        # Preprocess and interpolate
        rr_seconds, time_points, instantaneous_hr = self.preprocess_rr_intervals(rr_intervals)
        time_interp, hr_interp = self.interpolate_signal(time_points, instantaneous_hr, sampling_rate)

        if len(time_interp) < 2:
            return {"error": "Insufficient data for spectral analysis"}

        # Perform spectral analysis
        f, pxx = welch(hr_interp, fs=sampling_rate,
                      nperseg=min(len(hr_interp), 256),
                      scaling='density')

        # Analyze different frequency bands
        vlf_idx = (f >= 0.0033) & (f <= 0.04)  # Very low frequency (0.2-2.4 BPM)
        lf_idx = (f >= 0.04) & (f <= 0.15)    # Low frequency (2.4-9 BPM)
        rf_idx = (f >= 0.15) & (f <= 0.5)     # Respiratory frequency (9-30 BPM)

        spectral_details = {
            "method": self.name,
            "sampling_rate": sampling_rate,
            "signal_length": len(hr_interp),
            "frequency_range": [f[0], f[-1]],
            "frequency_resolution": f[1] - f[0],
            "frequency_bands": {}
        }

        # Analyze each frequency band
        bands_info = [
            ("VLF", vlf_idx, "Very Low Frequency (0.2-2.4 BPM)"),
            ("LF", lf_idx, "Low Frequency (2.4-9 BPM)"),
            ("RF", rf_idx, "Respiratory Frequency (9-30 BPM)")
        ]

        for band_name, band_idx, band_desc in bands_info:
            if np.any(band_idx):
                band_freqs = f[band_idx]
                band_powers = pxx[band_idx]
                peak_idx = np.argmax(band_powers)
                peak_freq = band_freqs[peak_idx]
                peak_power = band_powers[peak_idx]

                spectral_details["frequency_bands"][band_name] = {
                    "description": band_desc,
                    "frequency_range": [band_freqs[0], band_freqs[-1]],
                    "peak_frequency": peak_freq,
                    "peak_power": peak_power,
                    "equivalent_bpm": peak_freq * 60,
                    "power_ratio": peak_power / np.max(pxx) if np.max(pxx) > 0 else 0
                }

        # Determine selected peak
        breathing_rate = self.estimate(rr_intervals, sampling_rate)
        spectral_details["selected_breathing_rate"] = breathing_rate
        spectral_details["selection_logic"] = self._explain_peak_selection(rr_intervals, sampling_rate)

        return spectral_details

    def _explain_peak_selection(self, rr_intervals, sampling_rate):
        """
        Explain the logic behind peak selection.

        Args:
            rr_intervals (list): RR intervals
            sampling_rate (float): Sampling rate

        Returns:
            str: Explanation of peak selection
        """
        rr_seconds, time_points, instantaneous_hr = self.preprocess_rr_intervals(rr_intervals)
        time_interp, hr_interp = self.interpolate_signal(time_points, instantaneous_hr, sampling_rate)

        if len(time_interp) < 2:
            return "Insufficient data"

        f, pxx = welch(hr_interp, fs=sampling_rate,
                      nperseg=min(len(hr_interp), 256))

        # Check physiological range
        physio_idx = (f >= 0.2) & (f <= 0.4)
        resp_idx = (f >= 0.1) & (f <= 0.5)

        if np.any(physio_idx):
            return "Selected peak in physiological respiratory range (0.2-0.4 Hz = 12-24 BPM)"
        elif np.any(resp_idx):
            return "Selected peak in respiratory frequency band (0.1-0.5 Hz) - no physiological peaks found"
        else:
            return "No peaks found in respiratory frequency band"
