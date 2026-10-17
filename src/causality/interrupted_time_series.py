"""
Interrupted Time Series (ITS) Quasi-Experimental Causal Inference.
Evaluates public health interventions using segmented regression with Newey-West HAC robust inference.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


class InterruptedTimeSeriesModel:
    """
    Segmented regression model for Interrupted Time Series (ITS) quasi-experiments.
    Estimates immediate level shift (beta_2) and sustained slope change (beta_3)
    relative to counterfactual pre-intervention trajectories.
    """

    def __init__(
        self,
        outcome_var: str,
        intervention_cycle: int = 3,
        time_var: str = "cycle_index",
    ):
        self.outcome_var = outcome_var
        self.intervention_cycle = intervention_cycle
        self.time_var = time_var
        self.model_results = None
        self.data = None

    def prepare_design_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates ITS predictors:
          - T: continuous time index from 0 to N-1
          - D: binary intervention indicator (1 if post-intervention, 0 otherwise)
          - P: time elapsed post-intervention (0 before, 1, 2, ... after)
        """
        data = df.copy().sort_values(by=self.time_var).reset_index(drop=True)
        t_values = np.arange(len(data))
        d_values = (data[self.time_var] >= self.intervention_cycle).astype(int)

        # Time post-intervention
        p_values = np.zeros(len(data))
        post_indices = np.where(d_values == 1)[0]
        if len(post_indices) > 0:
            first_post_idx = post_indices[0]
            p_values[first_post_idx:] = np.arange(1, len(post_indices) + 1)

        data["T_time"] = t_values
        data["D_intervention"] = d_values
        data["P_post_trend"] = p_values
        return data

    def fit(self, df: pd.DataFrame, hac_lags: int = 1) -> Dict[str, Union[float, bool, pd.DataFrame]]:
        """
        Fits segmented regression with Newey-West HAC robust standard errors.
        """
        self.data = self.prepare_design_matrix(df)
        X = self.data[["T_time", "D_intervention", "P_post_trend"]]
        X = sm.add_constant(X)
        y = self.data[self.outcome_var]

        # OLS with HAC (Newey-West) standard errors
        ols = sm.OLS(y, X)
        self.model_results = ols.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

        # Counterfactual: What would have happened without intervention? (D=0, P=0)
        X_counterfactual = X.copy()
        X_counterfactual["D_intervention"] = 0
        X_counterfactual["P_post_trend"] = 0

        self.data["y_factual_pred"] = self.model_results.predict(X)
        self.data["y_counterfactual"] = self.model_results.predict(X_counterfactual)
        self.data["causal_effect"] = self.data["y_factual_pred"] - self.data["y_counterfactual"]

        # Parameters
        params = self.model_results.params
        bse = self.model_results.bse
        pvalues = self.model_results.pvalues

        summary_metrics = {
            "baseline_intercept_b0": round(float(params["const"]), 4),
            "baseline_slope_b1": round(float(params["T_time"]), 4),
            "baseline_slope_pvalue": round(float(pvalues["T_time"]), 4),
            "immediate_level_change_b2": round(float(params["D_intervention"]), 4),
            "level_change_se": round(float(bse["D_intervention"]), 4),
            "level_change_pvalue": round(float(pvalues["D_intervention"]), 4),
            "level_change_significant": bool(pvalues["D_intervention"] < config.SIGNIFICANCE_ALPHA),
            "slope_change_b3": round(float(params["P_post_trend"]), 4),
            "slope_change_se": round(float(bse["P_post_trend"]), 4),
            "slope_change_pvalue": round(float(pvalues["P_post_trend"]), 4),
            "slope_change_significant": bool(pvalues["P_post_trend"] < config.SIGNIFICANCE_ALPHA),
            "r_squared": round(float(self.model_results.rsquared), 4),
            "f_pvalue": round(float(self.model_results.f_pvalue), 4),
        }

        # Calculate final-period treatment effect
        last_row = self.data.iloc[-1]
        summary_metrics["final_period_treatment_effect"] = round(float(last_row["causal_effect"]), 4)
        summary_metrics["final_period_factual"] = round(float(last_row[self.outcome_var]), 4)
        summary_metrics["final_period_counterfactual"] = round(float(last_row["y_counterfactual"]), 4)

        if summary_metrics["final_period_counterfactual"] != 0:
            rel_effect = (
                summary_metrics["final_period_treatment_effect"]
                / summary_metrics["final_period_counterfactual"]
            ) * 100.0
            summary_metrics["final_period_relative_effect_pct"] = round(float(rel_effect), 2)
        else:
            summary_metrics["final_period_relative_effect_pct"] = 0.0

        return summary_metrics

    def get_projection_dataframe(self) -> pd.DataFrame:
        """Returns the dataframe containing actual, fitted, and counterfactual values."""
        if self.data is None:
            raise RuntimeError("Model must be fitted before retrieving projections.")
        return self.data[
            [
                self.time_var,
                "T_time",
                "D_intervention",
                self.outcome_var,
                "y_factual_pred",
                "y_counterfactual",
                "causal_effect",
            ]
        ]
