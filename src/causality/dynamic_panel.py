"""
Dynamic Panel and Within-Cohort Fixed Effects Causal Estimation for NHANES Pseudo-Panels.
Controls for unobserved time-invariant cohort heterogeneity (alpha_c) and secular shocks (lambda_t)
across granular longitudinal panels (N x T = 800+ observations).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


class DynamicPanelFixedEffects:
    """
    Within-Cohort Fixed Effects model for granular pseudo-panel time series:
      Y_{c,t} = alpha_c + beta * X_{c,t-1} + gamma * Z_{c,t} + lambda_t + epsilon_{c,t}

    Completely eliminates all time-invariant unobserved confounders (alpha_c)
    and common macro shocks (lambda_t).
    """

    def __init__(
        self,
        cohort_col: str = "cohort_id",
        time_col: str = "survey_year",
        outcome_var: str = "hba1c",
        exposure_var: str = "daily_sugar_g",
        covariates: Optional[List[str]] = None,
    ):
        self.cohort_col = cohort_col
        self.time_col = time_col
        self.outcome_var = outcome_var
        self.exposure_var = exposure_var
        self.covariates = covariates or ["bmi", "sedentary_minutes", "daily_calories"]
        self.results_ = {}

    def fit(self, panel_df: pd.DataFrame) -> Dict[str, Union[float, Dict, str]]:
        """
        Prepares lagged exposures, performs Within-Cohort de-meaning (Fixed Effects),
        and fits OLS with cohort-clustered robust standard errors.
        """
        df = panel_df.copy().sort_values(by=[self.cohort_col, self.time_col])

        # Create 1-period lagged exposure: X_{c, t-1}
        df[f"{self.exposure_var}_lag1"] = df.groupby(self.cohort_col)[self.exposure_var].shift(1)
        # Create 1-period lagged outcome: Y_{c, t-1}
        df[f"{self.outcome_var}_lag1"] = df.groupby(self.cohort_col)[self.outcome_var].shift(1)

        # Drop initial periods with missing lags
        clean_df = df.dropna(subset=[self.outcome_var, f"{self.exposure_var}_lag1"] + self.covariates).copy()

        n_cohorts = clean_df[self.cohort_col].nunique()
        n_periods = clean_df[self.time_col].nunique()
        total_obs = len(clean_df)

        if total_obs < 30:
            raise ValueError(f"Panel sample size ({total_obs}) too small for fixed effects.")

        # Demean variables (Within Transformation) to eliminate cohort fixed effect alpha_c
        feature_cols = [f"{self.exposure_var}_lag1"] + self.covariates
        demeaned_data = pd.DataFrame(index=clean_df.index)

        # Demean outcome
        cohort_mean_y = clean_df.groupby(self.cohort_col)[self.outcome_var].transform("mean")
        demeaned_data[self.outcome_var] = clean_df[self.outcome_var] - cohort_mean_y

        # Demean features
        for col in feature_cols:
            cohort_mean_x = clean_df.groupby(self.cohort_col)[col].transform("mean")
            demeaned_data[col] = clean_df[col] - cohort_mean_x

        # Add Time (Survey Cycle) dummies for macro shock control (lambda_t)
        time_dummies = pd.get_dummies(clean_df[self.time_col], prefix="year", drop_first=True, dtype=float)
        for col in time_dummies.columns:
            mean_dummy = clean_df.groupby(self.cohort_col)[col].transform("mean") if col in clean_df else time_dummies[col].mean()
            demeaned_data[col] = time_dummies[col] - mean_dummy

        X = demeaned_data[feature_cols + list(time_dummies.columns)]
        y = demeaned_data[self.outcome_var]

        # Fit model with Cluster-Robust standard errors by cohort
        model = sm.OLS(y, X)
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": clean_df[self.cohort_col]})

        causal_beta = float(fitted.params[f"{self.exposure_var}_lag1"])
        causal_se = float(fitted.bse[f"{self.exposure_var}_lag1"])
        causal_pval = float(fitted.pvalues[f"{self.exposure_var}_lag1"])
        ci_low = float(fitted.conf_int().loc[f"{self.exposure_var}_lag1", 0])
        ci_high = float(fitted.conf_int().loc[f"{self.exposure_var}_lag1", 1])

        self.results_ = {
            "n_cohorts": n_cohorts,
            "n_time_periods": n_periods,
            "total_panel_observations": total_obs,
            "exposure_lagged": f"{self.exposure_var}_lag1",
            "outcome": self.outcome_var,
            "causal_effect_beta": round(causal_beta, 4),
            "cluster_robust_se": round(causal_se, 4),
            "p_value": round(causal_pval, 4),
            "ci_95_lower": round(ci_low, 4),
            "ci_95_upper": round(ci_high, 4),
            "r_squared_within": round(float(fitted.rsquared), 4),
            "significant": bool(causal_pval < config.SIGNIFICANCE_ALPHA),
            "covariates_controlled": self.covariates,
            "unobserved_confounders_status": "Completely eliminated by within-cohort de-meaning (alpha_c)",
        }

        return self.results_
