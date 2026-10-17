"""
Data module for NHANES ingestion, synthetic cohort generation, and time series preparation.
"""

from .nhanes_downloader import NHANESDownloader
from .synthetic_generator import generate_nhanes_longitudinal_dataset
from .pseudo_panel import (
    build_pseudo_panel_cohorts,
    build_granular_pseudo_panel,
    aggregate_to_macro_timeseries,
)
from .accelerometer_loader import generate_accelerometer_timeseries

__all__ = [
    "NHANESDownloader",
    "generate_nhanes_longitudinal_dataset",
    "build_pseudo_panel_cohorts",
    "build_granular_pseudo_panel",
    "aggregate_to_macro_timeseries",
    "generate_accelerometer_timeseries",
]
