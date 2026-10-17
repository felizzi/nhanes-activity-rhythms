"""
Granger Causality and Stationarity Analysis Engine for Longitudinal Time Series.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

import config


def check_stationarity(
    series: pd.Series,
    alpha: float = config.SIGNIFICANCE_ALPHA,
) -> Dict[str, Union[float, bool, str]]:
    """
    Performs Augmented Dickey-Fuller (ADF) and KPSS tests to assess stationarity.
    - ADF Null Hypothesis: Series has a unit root (non-stationary).
    - KPSS Null Hypothesis: Series is stationary around a constant/trend.
    """
    clean_series = series.dropna()
    if len(clean_series) < 6:
        raise ValueError("Series is too short for stationarity testing (min 6 observations required).")

    # ADF Test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_result = adfuller(clean_series, autolag="AIC")
        adf_stat = float(adf_result[0])
        adf_pvalue = float(adf_result[1])
        is_stationary_adf = bool(adf_pvalue < alpha)

        # KPSS Test
        try:
            kpss_result = kpss(clean_series, regression="c", nlags="auto")
            kpss_stat = float(kpss_result[0])
            kpss_pvalue = float(kpss_result[1])
            is_stationary_kpss = bool(kpss_pvalue >= alpha)
        except Exception:
            kpss_stat = np.nan
            kpss_pvalue = np.nan
            is_stationary_kpss = is_stationary_adf

    # Combined interpretation
    if is_stationary_adf and is_stationary_kpss:
        conclusion = "Stationary (ADF rejects unit root; KPSS fails to reject stationarity)"
        stationary = True
    elif is_stationary_adf and not is_stationary_kpss:
        conclusion = "Difference/Trend Stationary"
        stationary = True
    elif not is_stationary_adf and is_stationary_kpss:
        conclusion = "Non-stationary (Near unit root)"
        stationary = False
    else:
        conclusion = "Non-stationary (Unit root confirmed by both tests)"
        stationary = False

    return {
        "variable": series.name or "series",
        "adf_stat": round(adf_stat, 4),
        "adf_pvalue": round(adf_pvalue, 4),
        "kpss_stat": round(kpss_stat, 4) if not np.isnan(kpss_stat) else None,
        "kpss_pvalue": round(kpss_pvalue, 4) if not np.isnan(kpss_pvalue) else None,
        "is_stationary": stationary,
        "conclusion": conclusion,
    }


def make_stationary(
    df: pd.DataFrame,
    variables: List[str],
    differencing_order: int = 1,
) -> pd.DataFrame:
    """
    Applies first-order differencing to render non-stationary time series I(0) stationary.
    Preserves original columns as well as diff_ columns.
    """
    diff_df = df.copy()
    for col in variables:
        if col in df.columns:
            diff_df[f"d_{col}"] = df[col].diff(periods=differencing_order)
    return diff_df.dropna().reset_index(drop=True)


def eval_granger_causality(
    data: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    max_lag: int = 2,
    test_statistic: str = "ssr_ftest",
    alpha: float = config.SIGNIFICANCE_ALPHA,
) -> Dict[str, Union[int, float, bool, str, Dict]]:
    """
    Tests if `cause_var` Granger-causes `effect_var`.
    In statsmodels: grangercausalitytests takes a 2D array [effect_var, cause_var]
    and tests H0: cause_var does NOT Granger-cause effect_var.
    """
    subset = data[[effect_var, cause_var]].dropna()
    n_obs = len(subset)

    if n_obs <= max_lag + 2:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for lag {max_lag}. Need at least {max_lag + 3}."
        )

    # Run Granger test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = grangercausalitytests(subset, maxlag=max_lag)

    best_lag = 1
    min_pvalue = 1.0
    lag_details = {}

    for lag in range(1, max_lag + 1):
        stat_tuple = results[lag][0][test_statistic]
        f_stat = float(stat_tuple[0])
        p_val = float(stat_tuple[1])

        lag_details[lag] = {
            "f_stat": round(f_stat, 4),
            "p_value": round(p_val, 4),
            "significant": bool(p_val < alpha),
        }

        if p_val < min_pvalue:
            min_pvalue = p_val
            best_lag = lag

    is_causal = bool(min_pvalue < alpha)

    return {
        "cause": cause_var,
        "effect": effect_var,
        "best_lag": best_lag,
        "best_pvalue": round(min_pvalue, 4),
        "is_causal": is_causal,
        "test_statistic": test_statistic,
        "lag_details": lag_details,
        "interpretation": (
            f"'{cause_var}' Granger-causes '{effect_var}' (p={min_pvalue:.4f} at lag {best_lag})"
            if is_causal
            else f"No evidence that '{cause_var}' Granger-causes '{effect_var}' (p={min_pvalue:.4f})"
        ),
    }


def eval_bidirectional_causality(
    data: pd.DataFrame,
    var_a: str,
    var_b: str,
    max_lag: int = 2,
    alpha: float = config.SIGNIFICANCE_ALPHA,
) -> Dict[str, Union[str, Dict]]:
    """
    Tests Granger causality in both directions (A -> B and B -> A)
    to differentiate between unidirectional causality, reverse causality, and feedback loops.
    """
    a_causes_b = eval_granger_causality(data, cause_var=var_a, effect_var=var_b, max_lag=max_lag, alpha=alpha)
    b_causes_a = eval_granger_causality(data, cause_var=var_b, effect_var=var_a, max_lag=max_lag, alpha=alpha)

    sig_a_b = a_causes_b["is_causal"]
    sig_b_a = b_causes_a["is_causal"]

    if sig_a_b and not sig_b_a:
        causal_direction = f"Unidirectional ({var_a} -> {var_b})"
    elif not sig_a_b and sig_b_a:
        causal_direction = f"Reverse Causality ({var_b} -> {var_a})"
    elif sig_a_b and sig_b_a:
        causal_direction = f"Bidirectional Feedback Loop ({var_a} <-> {var_b})"
    else:
        causal_direction = f"No Granger Causality detected ({var_a} _|_ {var_b})"

    return {
        "var_a": var_a,
        "var_b": var_b,
        "causal_direction": causal_direction,
        "a_to_b": a_causes_b,
        "b_to_a": b_causes_a,
    }


# Backwards compatibility aliases
test_granger_causality = eval_granger_causality
test_bidirectional_causality = eval_bidirectional_causality


def compute_pairwise_granger_matrix(
    data: pd.DataFrame,
    variables: List[str],
    max_lag: int = 2,
    alpha: float = config.SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """
    Computes an M x M matrix of Granger causality p-values where:
    Cell (row i, col j) represents the p-value for H0: variable_i does NOT Granger-cause variable_j.
    """
    p_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)

    for cause in variables:
        for effect in variables:
            if cause == effect:
                p_matrix.loc[cause, effect] = 1.0
            else:
                try:
                    res = test_granger_causality(data, cause_var=cause, effect_var=effect, max_lag=max_lag, alpha=alpha)
                    p_matrix.loc[cause, effect] = res["best_pvalue"]
                except Exception:
                    p_matrix.loc[cause, effect] = np.nan

    return p_matrix
