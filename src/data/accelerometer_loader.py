"""
NHANES Accelerometer / Physical Activity Monitor (PAM) Time Series Generator.
Simulates high-frequency intra-individual 7-day wearable time series with circadian rhythms
and dynamic behavioral feedbacks.
"""

from typing import Tuple
import numpy as np
import pandas as pd

import config


def generate_accelerometer_timeseries(
    n_days: int = 7,
    freq: str = "1h",
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a continuous wearable time series replicating NHANES Physical Activity Monitor data.
    Default frequency is 1-hour intervals (168 timestamps across 7 days).
    Captures:
      - Diurnal circadian rhythm in physical activity and metabolic expenditure.
      - Dynamic feedback: Intense physical activity in period t leads to compensatory rest
        and delayed elevated resting metabolic rate.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    dates = pd.date_range("2014-04-01 00:00:00", periods=n_hours, freq=freq)

    hour_of_day = dates.hour
    day_of_week = dates.dayofweek

    # Base circadian activity curve (peak around 14:00, nadir at 04:00)
    circadian_base = 250.0 + 350.0 * np.sin((hour_of_day - 8) / 24.0 * 2 * np.pi)
    circadian_base = np.maximum(circadian_base, 20.0)

    # Weekend effect: shifted peak, higher variance
    is_weekend = (day_of_week >= 5).astype(float)
    activity_mean = circadian_base * (1.0 + 0.15 * is_weekend)

    # Generate AR(1) autoregressive noise for realistic temporal autocorrelation
    phi = 0.55
    activity = np.zeros(n_hours)
    current_act = activity_mean[0]

    for t in range(n_hours):
        shock = rng.normal(0, 80.0)
        current_act = activity_mean[t] + phi * (current_act - activity_mean[max(0, t - 1)]) + shock
        activity[t] = max(10.0, current_act)

    # Sedentary minutes per hour (max 60 mins)
    # Higher activity strongly reduces sedentary time
    sedentary_mins = np.clip(60.0 - (activity / 600.0) * 45.0 + rng.normal(0, 4.0, size=n_hours), 5.0, 60.0)

    # Dynamic Metabolic Expenditure (METs / Estimated Caloric Burn per hour)
    # Causal lag: MET burn is driven by current activity + delayed excess post-exercise oxygen consumption (EPOC)
    mets = np.zeros(n_hours)
    mets[0] = 1.0 + activity[0] * 0.003
    for t in range(1, n_hours):
        # Current activity + 25% lagged activity carryover
        mets[t] = 1.0 + activity[t] * 0.0035 + activity[t - 1] * 0.0012 + rng.normal(0, 0.15)
        mets[t] = max(0.9, mets[t])

    # Estimated Heart Rate (bpm): baseline 65 bpm, responsive to activity with persistence
    hr = np.zeros(n_hours)
    hr[0] = 68.0
    for t in range(1, n_hours):
        target_hr = 62.0 + activity[t] * 0.075 + (mets[t] - 1.0) * 8.0
        hr[t] = 0.4 * hr[t - 1] + 0.6 * target_hr + rng.normal(0, 2.5)
        hr[t] = np.clip(hr[t], 45.0, 185.0)

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend.astype(int),
            "activity_counts": np.round(activity, 1),
            "sedentary_minutes": np.round(sedentary_mins, 1),
            "metabolic_rate_mets": np.round(mets, 2),
            "heart_rate_bpm": np.round(hr, 1),
        }
    )
    return df
