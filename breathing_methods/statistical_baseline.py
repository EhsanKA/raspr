#!/usr/bin/env python3
"""
Statistical Baseline Breathing Rate Estimator

This module implements a simple statistical approach to breathing rate estimation
based on the relationship between heart rate and breathing rate. It uses empirical
rules and statistical measures of RR intervals.

Author: Modular breathing rate analysis
Date: September 2025
"""

import numpy as np
from .base_estimator import BreathingRateEstimator, HRVFeatures


class StatisticalEstimator(BreathingRateEstimator):
    """
    Breathing rate estimation using statistical methods.

    This method uses simple statistical relationships between heart rate,
    RR interval characteristics, and breathing rate. It's designed to provide
    a baseline comparison for more sophisticated methods.
    """

    def __init__(self):
        super().__init__(
            name="Statistical Baseline",
            description="Estimates breathing rate using statistical relationships and empirical rules"
        )

    def estimate(self, rr_intervals, sampling_rate=4.0):
        """
        Estimate breathing rate using statistical methods.

        Args:
            rr_intervals (list): List of RR intervals in milliseconds
            sampling_rate (float): Sampling rate (not used in this method)

        Returns:
            float: Estimated breathing rate in BPM, or np.nan if estimation fails
        """
        # Validate input
        if not self.validate_input(rr_intervals):
            return np.nan

        # Extract HRV features
        hrv_features = HRVFeatures.extract_all_features(rr_intervals)

        # Estimate breathing rate using statistical relationships
        breathing_rate = self._estimate_from_statistics(hrv_features)

        return breathing_rate

    def _estimate_from_statistics(self, hrv_features):
        """
        Estimate breathing rate using statistical relationships.

        Args:
            hrv_features (dict): Dictionary of HRV features

        Returns:
            float: Estimated breathing rate in BPM
        """
        mean_rr = hrv_features['mean_rr']
        cv_rr = hrv_features['cv_rr']

        # Method 1: RR interval-based estimation
        # Longer RR intervals generally indicate slower breathing
        if mean_rr > 900:  # Very long RR intervals
            rr_based_br = 10.0
        elif mean_rr > 800:  # Long RR intervals
            rr_based_br = 12.0
        elif mean_rr > 700:  # Normal RR intervals
            rr_based_br = 15.0
        elif mean_rr > 600:  # Short RR intervals
            rr_based_br = 18.0
        else:  # Very short RR intervals
            rr_based_br = 20.0

        # Method 2: Variability-based adjustment
        # Higher variability might indicate more irregular breathing
        if cv_rr > 5:
            variability_factor = 1.1  # Slightly faster breathing
        elif cv_rr > 3:
            variability_factor = 1.0  # Normal
        else:
            variability_factor = 0.9  # More regular, slightly slower

        # Method 3: Heart rate-based estimation (if we can derive it)
        # Use the relationship: BR ≈ HR/4 for rough estimation
        if mean_rr > 0:
            estimated_hr = 60000 / mean_rr  # Convert RR to HR
            hr_based_br = estimated_hr / 4  # Rough BR estimation
            hr_based_br = max(10.0, min(20.0, hr_based_br))  # Constrain
        else:
            hr_based_br = 15.0

        # Combine methods with weights
        combined_br = (rr_based_br * 0.5 + hr_based_br * 0.3 + rr_based_br * variability_factor * 0.2)

        # Constrain to physiological range
        breathing_rate = max(10.0, min(20.0, combined_br))

        return round(breathing_rate, 1)

    def get_statistical_details(self, rr_intervals):
        """
        Get detailed statistical analysis for debugging.

        Args:
            rr_intervals (list): RR intervals in milliseconds

        Returns:
            dict: Detailed statistical analysis
        """
        if not self.validate_input(rr_intervals):
            return {"error": "Invalid input"}

        hrv_features = HRVFeatures.extract_all_features(rr_intervals)
        breathing_rate = self.estimate(rr_intervals)

        mean_rr = hrv_features['mean_rr']
        cv_rr = hrv_features['cv_rr']

        # Calculate individual method contributions
        if mean_rr > 900:
            rr_based_br = 10.0
        elif mean_rr > 800:
            rr_based_br = 12.0
        elif mean_rr > 700:
            rr_based_br = 15.0
        elif mean_rr > 600:
            rr_based_br = 18.0
        else:
            rr_based_br = 20.0

        if cv_rr > 5:
            variability_factor = 1.1
        elif cv_rr > 3:
            variability_factor = 1.0
        else:
            variability_factor = 0.9

        if mean_rr > 0:
            estimated_hr = 60000 / mean_rr
            hr_based_br = max(10.0, min(20.0, estimated_hr / 4))
        else:
            hr_based_br = 15.0

        combined_br = (rr_based_br * 0.5 + hr_based_br * 0.3 + rr_based_br * variability_factor * 0.2)

        return {
            "method": self.name,
            "hrv_features": hrv_features,
            "method_contributions": {
                "rr_interval_method": {
                    "mean_rr": mean_rr,
                    "category": self._categorize_rr_interval(mean_rr),
                    "breathing_rate": rr_based_br
                },
                "heart_rate_method": {
                    "estimated_hr": estimated_hr if mean_rr > 0 else 0,
                    "breathing_rate": hr_based_br
                },
                "variability_method": {
                    "cv_rr": cv_rr,
                    "factor": variability_factor,
                    "adjusted_br": rr_based_br * variability_factor
                }
            },
            "weights": {
                "rr_interval_weight": 0.5,
                "heart_rate_weight": 0.3,
                "variability_weight": 0.2
            },
            "intermediate_calculation": combined_br,
            "final_breathing_rate": breathing_rate,
            "physiological_range": "10-20 BPM (baseline)",
            "reliability_assessment": self._assess_reliability(hrv_features)
        }

    def _categorize_rr_interval(self, mean_rr):
        """
        Categorize RR interval for interpretation.

        Args:
            mean_rr (float): Mean RR interval in ms

        Returns:
            str: Category description
        """
        if mean_rr > 900:
            return "Very long (>900ms) - likely slow breathing"
        elif mean_rr > 800:
            return "Long (800-900ms) - slower breathing"
        elif mean_rr > 700:
            return "Normal (700-800ms) - normal breathing"
        elif mean_rr > 600:
            return "Short (600-700ms) - faster breathing"
        else:
            return "Very short (<600ms) - rapid breathing"

    def _assess_reliability(self, hrv_features):
        """
        Assess the reliability of the statistical estimation.

        Args:
            hrv_features (dict): HRV features

        Returns:
            str: Reliability assessment
        """
        count = hrv_features['count']
        cv_rr = hrv_features['cv_rr']

        if count < 10:
            return "Low reliability - insufficient data points"
        elif cv_rr > 10:
            return "Moderate reliability - high variability may affect accuracy"
        elif cv_rr < 2:
            return "High reliability - low variability, consistent signal"
        else:
            return "Good reliability - normal variability range"
