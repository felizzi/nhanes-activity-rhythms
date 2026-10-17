"""
Vector Autoregression (VAR), Causal Impulse Response Functions (IRF),
and Forecast Error Variance Decomposition (FEVD) for NHANES Time Series.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

import config


def fit_causal_var_model(
    df: pd.DataFrame,
    variables: List[str],
    max_lags: int = config.MAX_VAR_LAGS,
    ic: str = "aic",
) -> Tuple[object, Dict[str, Union[int, float, Dict]]]:
    """
    Fits a Vector Autoregressive (VAR) model to multi-wave longitudinal data.
    Automates lag selection using Information Criteria (AIC/BIC).
    """
    var_data = df[variables].dropna()
    if len(var_data) < max_lags * len(variables) + 2:
        # Fall back to 1 lag if sample size is constrained
        max_lags = 1

    model = VAR(var_data)
    selected_lag = 1
    try:
        if len(var_data) >= (len(variables) * max_lags + len(variables) + 3):
            lag_order_results = model.select_order(maxlags=max_lags)
            ic_lag = getattr(lag_order_results, ic)
            if ic_lag is not None and ic_lag > 0:
                selected_lag = ic_lag
    except Exception:
        selected_lag = 1

    fitted_model = model.fit(maxlags=selected_lag, ic=None)

    # Stability diagnostic: all companion matrix roots must be strictly inside the unit circle
    roots = fitted_model.roots
    is_stable = bool(np.all(np.abs(roots) > 1.0))

    try:
        aic_val = float(fitted_model.aic)
        bic_val = float(fitted_model.bic)
        hqic_val = float(fitted_model.hqic)
    except Exception:
        aic_val = np.nan
        bic_val = np.nan
        hqic_val = np.nan

    diagnostics = {
        "selected_lag": int(selected_lag),
        "aic": aic_val,
        "bic": bic_val,
        "hqic": hqic_val,
        "is_stable": is_stable,
        "n_obs": int(fitted_model.nobs),
        "variables": variables,
    }

    return fitted_model, diagnostics


def compute_irf_and_fevd(
    fitted_var_model,
    periods: int = config.FORECAST_HORIZON,
    orth: bool = True,
) -> Dict[str, Union[pd.DataFrame, object]]:
    """
    Computes:
      1. Orthogonalized Impulse Response Functions (IRF) with asymptotic confidence intervals.
      2. Forecast Error Variance Decomposition (FEVD) over the forecast horizon.
    """
    irf_result = fitted_var_model.irf(periods=periods)
    fevd_result = fitted_var_model.fevd(periods=periods)

    variables = fitted_var_model.names
    irf_matrices = irf_result.orth_irfs if orth else irf_result.irfs

    # Build structured summary table for the terminal response
    terminal_impacts = {}
    for i, cause in enumerate(variables):
        for j, effect in enumerate(variables):
            # Cumulative response across horizons
            cum_irf = irf_result.cum_effects[periods, j, i]
            terminal_impacts[f"shock_{cause} -> {effect}"] = round(float(cum_irf), 4)

    return {
        "irf_obj": irf_result,
        "fevd_obj": fevd_result,
        "variables": variables,
        "periods": periods,
        "cumulative_effects_horizon": terminal_impacts,
    }
