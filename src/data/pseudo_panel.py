"""
Deaton Pseudo-Panel Cohort Builder and Macro Time Series Aggregator.
Transforms repeated cross-sectional NHANES surveys into longitudinal panel time series.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Computes survey-weighted mean, dropping NaNs."""
    valid = (~series.isna()) & (~weights.isna())
    if not valid.any():
        return np.nan
    return float(np.average(series[valid], weights=weights[valid]))


def build_pseudo_panel_cohorts(
    df: pd.DataFrame,
    cohort_col: str = "cohort",
    time_col: str = "survey_year",
    weight_col: str = "exam_weight",
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Constructs a longitudinal pseudo-panel tracking demographic/birth cohorts across survey cycles.
    Computes survey-weighted means for each cohort-time cell: (c, t).
    """
    if variables is None:
        variables = [
            "daily_calories",
            "daily_sugar_g",
            "sedentary_minutes",
            "bmi",
            "waist_circumference",
            "hba1c",
            "systolic_bp",
            "diastolic_bp",
            "total_cholesterol",
            "hdl_cholesterol",
        ]

    cohort_records = []
    cohorts = sorted(df[cohort_col].unique())
    time_points = sorted(df[time_col].unique())

    for c in cohorts:
        for t in time_points:
            cell_data = df[(df[cohort_col] == c) & (df[time_col] == t)]
            if len(cell_data) == 0:
                continue

            weights = cell_data[weight_col] if weight_col in cell_data.columns else pd.Series(1, index=cell_data.index)
            row = {
                cohort_col: c,
                time_col: t,
                "sample_size": len(cell_data),
            }

            if "cycle_index" in cell_data.columns:
                row["cycle_index"] = cell_data["cycle_index"].iloc[0]
            if "policy_intervention" in cell_data.columns:
                row["policy_intervention"] = cell_data["policy_intervention"].iloc[0]

            for var in variables:
                if var in cell_data.columns:
                    row[var] = weighted_mean(cell_data[var], weights)

            cohort_records.append(row)

    pseudo_df = pd.DataFrame(cohort_records)
    pseudo_df = pseudo_df.sort_values(by=[cohort_col, time_col]).reset_index(drop=True)
    return pseudo_df


def aggregate_to_macro_timeseries(
    df: pd.DataFrame,
    time_col: str = "survey_year",
    weight_col: str = "exam_weight",
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregates multi-cycle NHANES microdata into a single national population-level time series
    indexed by survey wave / year using survey weights.
    """
    if variables is None:
        variables = [
            "daily_calories",
            "daily_sugar_g",
            "sedentary_minutes",
            "bmi",
            "waist_circumference",
            "hba1c",
            "systolic_bp",
            "diastolic_bp",
            "total_cholesterol",
            "hdl_cholesterol",
        ]

    time_records = []
    time_points = sorted(df[time_col].unique())

    for t in time_points:
        cell_data = df[df[time_col] == t]
        weights = cell_data[weight_col] if weight_col in cell_data.columns else pd.Series(1, index=cell_data.index)

        row = {
            time_col: t,
            "sample_size": len(cell_data),
        }

        if "cycle_index" in cell_data.columns:
            row["cycle_index"] = cell_data["cycle_index"].iloc[0]
        if "policy_intervention" in cell_data.columns:
            row["policy_intervention"] = cell_data["policy_intervention"].iloc[0]

        for var in variables:
            if var in cell_data.columns:
                row[var] = weighted_mean(cell_data[var], weights)

        time_records.append(row)

    ts_df = pd.DataFrame(time_records)
    ts_df = ts_df.sort_values(by=time_col).reset_index(drop=True)
    return ts_df


def build_granular_pseudo_panel(
    df: pd.DataFrame,
    cohort_col: str = "granular_cohort",
    time_col: str = "survey_year",
    min_cell_size: int = 8,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Builds a large-scale granular pseudo-panel across 5-year birth brackets x Gender x Race.
    Yields hundreds of longitudinal cohort trajectories (N x T = 500-1,000+ observations).
    Filters out sparse cells with sample size < min_cell_size.
    """
    raw_panel = build_pseudo_panel_cohorts(
        df=df,
        cohort_col=cohort_col,
        time_col=time_col,
        variables=variables,
    )
    # Filter cells with sufficient sample size
    filtered_panel = raw_panel[raw_panel["sample_size"] >= min_cell_size].copy()
    # Filter cohorts observed in at least 4 cycles to ensure dynamic lag viability
    cohort_counts = filtered_panel.groupby(cohort_col)[time_col].count()
    valid_cohorts = cohort_counts[cohort_counts >= 4].index
    final_panel = filtered_panel[filtered_panel[cohort_col].isin(valid_cohorts)].reset_index(drop=True)
    return final_panel
