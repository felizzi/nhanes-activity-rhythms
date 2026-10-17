"""
Potential Outcomes Framework and Doubly Robust Causal Inference for NHANES.
Implements Augmented Inverse Probability Weighting (AIPW), Marginal Structural Models (MSM)
with Stabilized IPTW, Covariate Balance Diagnostics, and VanderWeele's E-value Sensitivity Analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

import config


class PotentialOutcomesEstimator:
    """
    Estimates the Average Treatment Effect (ATE) under the Potential Outcomes framework:
      ATE = E[Y(1) - Y(0)]
    Uses Doubly Robust Augmented Inverse Probability Weighting (AIPW) and
    Marginal Structural Models with Stabilized IPTW.
    """

    def __init__(
        self,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
        clip_bounds: Tuple[float, float] = (0.02, 0.98),
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounder_cols = confounder_cols
        self.clip_bounds = clip_bounds
        self.results_ = {}

    def fit(self, df: pd.DataFrame) -> Dict[str, Union[float, pd.DataFrame, Dict]]:
        """
        Fits the propensity score model, outcome models, computes weights,
        evaluates covariate balance, and estimates the Doubly Robust ATE.
        """
        clean_df = df.dropna(subset=[self.treatment_col, self.outcome_col] + self.confounder_cols).copy()
        n = len(clean_df)
        if n < 50:
            raise ValueError(f"Sample size too small ({n}) for robust potential outcomes estimation.")

        A = clean_df[self.treatment_col].values.astype(float)
        Y = clean_df[self.outcome_col].values.astype(float)
        X = clean_df[self.confounder_cols].values

        # -------------------------------------------------------------
        # 1. Propensity Score Estimation: P(A = 1 | X)
        # -------------------------------------------------------------
        propensity_model = LogisticRegression(max_iter=1000, penalty="l2", C=1.0)
        propensity_model.fit(X, A)
        ps = propensity_model.predict_proba(X)[:, 1]

        # Enforce positivity: Clip extreme propensity scores
        ps_clipped = np.clip(ps, self.clip_bounds[0], self.clip_bounds[1])

        # -------------------------------------------------------------
        # 2. Stabilized Inverse Probability Weights (IPTW)
        # -------------------------------------------------------------
        marginal_prob_treated = np.mean(A)
        stabilized_weights = np.where(
            A == 1,
            marginal_prob_treated / ps_clipped,
            (1.0 - marginal_prob_treated) / (1.0 - ps_clipped),
        )

        clean_df["propensity_score"] = ps_clipped
        clean_df["iptw"] = stabilized_weights

        # -------------------------------------------------------------
        # 3. Covariate Balance Diagnostics (Standardized Mean Difference)
        # -------------------------------------------------------------
        balance_records = []
        for j, col_name in enumerate(self.confounder_cols):
            x_var = X[:, j]

            # Unweighted SMD
            mean_1 = np.mean(x_var[A == 1])
            mean_0 = np.mean(x_var[A == 0])
            var_1 = np.var(x_var[A == 1], ddof=1)
            var_0 = np.var(x_var[A == 0], ddof=1)
            pooled_sd = np.sqrt((var_1 + var_0) / 2.0)
            smd_unweighted = abs(mean_1 - mean_0) / pooled_sd if pooled_sd > 0 else 0.0

            # Weighted SMD
            w1 = stabilized_weights[A == 1]
            w0 = stabilized_weights[A == 0]
            w_mean_1 = np.average(x_var[A == 1], weights=w1)
            w_mean_0 = np.average(x_var[A == 0], weights=w0)
            smd_weighted = abs(w_mean_1 - w_mean_0) / pooled_sd if pooled_sd > 0 else 0.0

            balance_records.append(
                {
                    "confounder": col_name,
                    "smd_unweighted": round(float(smd_unweighted), 4),
                    "smd_weighted": round(float(smd_weighted), 4),
                    "balanced": bool(smd_weighted < 0.10),
                }
            )
        balance_df = pd.DataFrame(balance_records)

        # -------------------------------------------------------------
        # 4. Naive Observational Association (Unadjusted)
        # -------------------------------------------------------------
        naive_treated_mean = np.mean(Y[A == 1])
        naive_control_mean = np.mean(Y[A == 0])
        naive_difference = naive_treated_mean - naive_control_mean

        # -------------------------------------------------------------
        # 5. Outcome Regression Models: mu_1(X) and mu_0(X)
        # -------------------------------------------------------------
        reg_1 = Ridge(alpha=1.0)
        reg_0 = Ridge(alpha=1.0)
        reg_1.fit(X[A == 1], Y[A == 1])
        reg_0.fit(X[A == 0], Y[A == 0])

        mu_1_pred = reg_1.predict(X)
        mu_0_pred = reg_0.predict(X)

        # -------------------------------------------------------------
        # 6. Doubly Robust Estimator (AIPW)
        # -------------------------------------------------------------
        # AIPW individual influence scores
        gamma_1 = mu_1_pred + (A * (Y - mu_1_pred)) / ps_clipped
        gamma_0 = mu_0_pred + ((1.0 - A) * (Y - mu_0_pred)) / (1.0 - ps_clipped)
        dr_individual_effects = gamma_1 - gamma_0

        ate_dr = float(np.mean(dr_individual_effects))
        ate_var = float(np.var(dr_individual_effects, ddof=1) / n)
        ate_se = float(np.sqrt(ate_var))
        t_stat = ate_dr / ate_se if ate_se > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        ci_lower = float(ate_dr - 1.96 * ate_se)
        ci_upper = float(ate_dr + 1.96 * ate_se)

        # -------------------------------------------------------------
        # 7. Marginal Structural Model (IPTW-weighted WLS)
        # -------------------------------------------------------------
        weighted_y1 = np.average(Y[A == 1], weights=stabilized_weights[A == 1])
        weighted_y0 = np.average(Y[A == 0], weights=stabilized_weights[A == 0])
        ate_msm = float(weighted_y1 - weighted_y0)

        # -------------------------------------------------------------
        # 8. VanderWeele's E-Value Sensitivity Analysis
        # -------------------------------------------------------------
        # For continuous outcome, approximate Risk Ratio: RR ~ exp(0.91 * d)
        sd_y = np.std(Y, ddof=1)
        cohen_d = ate_dr / sd_y if sd_y > 0 else 0.0
        rr_approx = np.exp(0.91 * abs(cohen_d))
        if rr_approx > 1.0:
            e_value = float(rr_approx + np.sqrt(rr_approx * (rr_approx - 1.0)))
        else:
            e_value = 1.0

        self.results_ = {
            "n_observations": n,
            "treatment": self.treatment_col,
            "outcome": self.outcome_col,
            "naive_observational_difference": round(float(naive_difference), 4),
            "ate_doubly_robust": round(ate_dr, 4),
            "ate_standard_error": round(ate_se, 4),
            "p_value": round(p_val, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "ate_msm_iptw": round(ate_msm, 4),
            "confounding_bias_removed": round(float(naive_difference - ate_dr), 4),
            "e_value": round(e_value, 2),
            "balance_df": balance_df,
            "mean_propensity_treated": round(float(np.mean(ps_clipped[A == 1])), 3),
            "mean_propensity_control": round(float(np.mean(ps_clipped[A == 0])), 3),
        }

        return self.results_
