"""
Unit and Integration Tests for NHANES Causal Time Series Analysis.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import generate_nhanes_longitudinal_dataset
from src.data.pseudo_panel import (
    build_pseudo_panel_cohorts,
    build_granular_pseudo_panel,
    aggregate_to_macro_timeseries,
)
from src.data.accelerometer_loader import generate_accelerometer_timeseries
from src.causality.potential_outcomes import PotentialOutcomesEstimator
from src.causality.dynamic_panel import DynamicPanelFixedEffects
from src.causality.convergent_cross_mapping import convergent_cross_mapping
from src.causality.granger_causality import (
    check_stationarity,
    make_stationary,
    eval_granger_causality,
    eval_bidirectional_causality,
    compute_pairwise_granger_matrix,
)
from src.causality.var_causal_irf import fit_causal_var_model, compute_irf_and_fevd
from src.causality.interrupted_time_series import InterruptedTimeSeriesModel
from src.causality.causal_dag import build_longitudinal_causal_dag, analyze_causal_paths


@pytest.fixture(scope="module")
def sample_dataset():
    """Generates a small test dataset across 10 standard survey cycles."""
    df = generate_nhanes_longitudinal_dataset(n_per_cycle=350, seed=123)
    return df


def test_synthetic_data_generation(sample_dataset):
    df = sample_dataset
    assert not df.empty
    assert len(df) == 350 * 10
    assert "daily_sugar_g" in df.columns
    assert "high_sugar_treatment" in df.columns
    assert "bmi" in df.columns
    assert "hba1c" in df.columns
    assert "granular_cohort" in df.columns
    assert df["bmi"].mean() > 20.0


def test_pseudo_panel_construction(sample_dataset):
    pseudo_df = build_pseudo_panel_cohorts(sample_dataset)
    assert not pseudo_df.empty
    assert "cohort" in pseudo_df.columns
    assert "survey_year" in pseudo_df.columns
    assert "daily_sugar_g" in pseudo_df.columns

    macro_df = aggregate_to_macro_timeseries(sample_dataset)
    assert len(macro_df) == 10
    assert "daily_sugar_g" in macro_df.columns

    granular_df = build_granular_pseudo_panel(sample_dataset, min_cell_size=5)
    assert len(granular_df) > 50


def test_potential_outcomes_aipw(sample_dataset):
    estimator = PotentialOutcomesEstimator(
        treatment_col="high_sugar_treatment",
        outcome_col="hba1c",
        confounder_cols=["age", "gender", "poverty_income_ratio", "sedentary_minutes"],
    )
    res = estimator.fit(sample_dataset)
    assert "ate_doubly_robust" in res
    assert "ate_standard_error" in res
    assert "e_value" in res
    assert res["ate_doubly_robust"] > 0
    assert res["e_value"] >= 1.0
    assert len(res["balance_df"]) == 4


def test_dynamic_panel_fixed_effects(sample_dataset):
    granular_df = build_granular_pseudo_panel(sample_dataset, min_cell_size=5)
    dp = DynamicPanelFixedEffects(
        cohort_col="granular_cohort",
        time_col="survey_year",
        outcome_var="hba1c",
        exposure_var="daily_sugar_g",
        covariates=["bmi", "sedentary_minutes"],
    )
    res = dp.fit(granular_df)
    assert "causal_effect_beta" in res
    assert "cluster_robust_se" in res
    assert res["n_cohorts"] > 10


def test_convergent_cross_mapping():
    accel_df = generate_accelerometer_timeseries(n_days=4, freq="1h", seed=42)
    ccm_res = convergent_cross_mapping(
        series_x=accel_df["activity_counts"],
        series_y=accel_df["metabolic_rate_mets"],
        embedding_dim=3,
        tau=1,
    )
    assert "rho_x_causes_y" in ccm_res
    assert len(ccm_res["rho_x_causes_y"]) > 1
    assert "causal_direction" in ccm_res


def test_stationarity_check(sample_dataset):
    macro_df = aggregate_to_macro_timeseries(sample_dataset)
    res = check_stationarity(macro_df["daily_sugar_g"])
    assert "adf_stat" in res
    assert "adf_pvalue" in res
    assert "is_stationary" in res
    assert isinstance(res["is_stationary"], bool)


def test_eval_granger_causality(sample_dataset):
    macro_df = aggregate_to_macro_timeseries(sample_dataset)
    res = eval_granger_causality(macro_df, cause_var="daily_sugar_g", effect_var="hba1c", max_lag=1)
    assert "best_lag" in res
    assert "best_pvalue" in res
    assert "is_causal" in res

    bidir = eval_bidirectional_causality(macro_df, "daily_sugar_g", "hba1c", max_lag=1)
    assert "causal_direction" in bidir


def test_var_causal_irf(sample_dataset):
    macro_df = aggregate_to_macro_timeseries(sample_dataset)
    vars_to_fit = ["daily_sugar_g", "bmi", "hba1c"]
    var_model, diag = fit_causal_var_model(macro_df, vars_to_fit, max_lags=1)
    assert var_model is not None
    assert diag["selected_lag"] >= 1

    irf_res = compute_irf_and_fevd(var_model, periods=4)
    assert "cumulative_effects_horizon" in irf_res
    assert len(irf_res["cumulative_effects_horizon"]) > 0


def test_interrupted_time_series(sample_dataset):
    macro_df = aggregate_to_macro_timeseries(sample_dataset)
    its = InterruptedTimeSeriesModel(outcome_var="daily_sugar_g", intervention_cycle=2)
    metrics = its.fit(macro_df)
    assert "immediate_level_change_b2" in metrics
    assert "slope_change_b3" in metrics
    assert "final_period_treatment_effect" in metrics


def test_causal_dag():
    dag = build_longitudinal_causal_dag()
    assert dag.number_of_nodes() > 5
    assert dag.number_of_edges() > 10

    paths = analyze_causal_paths(dag, source="Dietary_Sugar_(t-1)", target="HbA1c_(t)")
    assert len(paths["all_paths"]) >= 2
    assert "BMI_(t)" in paths["mediators"]
