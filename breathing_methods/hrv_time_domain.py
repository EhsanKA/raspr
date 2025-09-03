#!/usr/bin/env python3
"""
HRV Time-Domain Breathing Rate Estimator

This module implements breathing rate estimation using heart rate variability
features in the time domain. It uses RMSSD, SDNN, pNN50, and coefficient of
variation to estimate breathing rate based on respiratory sinus arrhythmia.

Author: Modular breathing rate analysis
Date: September 2025
"""

import numpy as np
from .base_estimator import BreathingRateEstimator, HRVFeatures


class HRVTimeDomainEstimator(BreathingRateEstimator):
    """
    Breathing rate estimation using HRV time-domain features.

    This method analyzes heart rate variability in the time domain to estimate
    breathing rate. It uses the relationship between respiratory sinus arrhythmia
    and breathing patterns.
    """

    def __init__(self):
        super().__init__(
            name="HRV Time-Domain",
            description="Estimates breathing rate using heart rate variability features in the time domain"
        )

    def estimate(self, rr_intervals, sampling_rate=4.0):
        """
        Estimate breathing rate using HRV time-domain analysis.

        Args:
            rr_intervals (list): List of RR intervals in milliseconds
            sampling_rate (float): Sampling rate for interpolation (Hz) - not used in this method

        Returns:
            float: Estimated breathing rate in BPM, or np.nan if estimation fails
        """
        # Validate input
        if not self.validate_input(rr_intervals):
            return np.nan

        # Extract HRV features
        hrv_features = HRVFeatures.extract_all_features(rr_intervals)

        # Estimate breathing rate using corrected relationship for resting conditions
        breathing_rate = self._estimate_from_hrv_features(hrv_features)

        return breathing_rate

    def _estimate_from_hrv_features(self, hrv_features):
        """
        Estimate breathing rate from HRV features using corrected relationship.

        Args:
            hrv_features (dict): Dictionary of HRV features

        Returns:
            float: Estimated breathing rate in BPM
        """
        rmssd = hrv_features['rmssd']
        cv_rr = hrv_features['cv_rr']
        pnn50 = hrv_features['pnn50']

        # Baseline breathing rate for resting conditions
        baseline_br = 16.0

        # Corrected RMSSD relationship for resting physiology
        # RMSSD values around 10-50 ms are normal for resting conditions
        if rmssd < 10:
            # Low RMSSD - slightly faster breathing (reduced vagal tone)
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
        # Lower CV indicates more regular breathing (closer to baseline)
        if cv_rr > 0:
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

        # Constrain to physiological range for resting conditions (8-25 breaths/min)
        breathing_rate = max(8.0, min(25.0, breathing_rate))

        return round(breathing_rate, 1)

    def get_feature_importance(self, rr_intervals):
        """
        Get detailed feature analysis for debugging.

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            dict: Detailed feature analysis
        """
        if not self.validate_input(rr_intervals):
            return {"error": "Invalid input"}

        hrv_features = HRVFeatures.extract_all_features(rr_intervals)
        breathing_rate = self._estimate_from_hrv_features(hrv_features)

        # Calculate factor contributions
        rmssd = hrv_features['rmssd']
        cv_rr = hrv_features['cv_rr']
        pnn50 = hrv_features['pnn50']
        baseline_br = 16.0

        if rmssd < 10:
            rmssd_factor = 1.1
        elif rmssd < 20:
            rmssd_factor = 1.0
        elif rmssd < 40:
            rmssd_factor = 0.9
        else:
            rmssd_factor = 0.8

        cv_factor = max(0.8, min(1.2, 1.0 - (cv_rr - 3.0) / 10.0)) if cv_rr > 0 else 1.0
        pnn50_factor = max(0.9, min(1.1, 1.0 + (pnn50 - 5.0) / 20.0)) if pnn50 > 0 else 1.0

        return {
            "method": self.name,
            "hrv_features": hrv_features,
            "factors": {
                "rmssd_factor": rmssd_factor,
                "cv_factor": cv_factor,
                "pnn50_factor": pnn50_factor,
                "baseline_br": baseline_br
            },
            "intermediate_calculation": baseline_br * rmssd_factor * cv_factor * pnn50_factor,
            "final_breathing_rate": breathing_rate,
            "physiological_range": "8-25 BPM (resting)",
            "rmssd_interpretation": self._interpret_rmssd(rmssd)
        }

    def _interpret_rmssd(self, rmssd):
        """
        Interpret RMSSD value for physiological context.

        Args:
            rmssd (float): RMSSD value

        Returns:
            str: Interpretation
        """
        if rmssd < 10:
            return "Low RMSSD (<10ms) - Reduced vagal tone, slightly faster breathing expected"
        elif rmssd < 20:
            return "Normal RMSSD (10-20ms) - Good respiratory modulation, normal breathing"
        elif rmssd < 40:
            return "High RMSSD (20-40ms) - Strong vagal tone, slightly slower breathing"
        else:
            return "Very high RMSSD (>40ms) - Excellent vagal tone, slower breathing"
