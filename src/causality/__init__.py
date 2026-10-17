"""
Causality module for time series and longitudinal analysis.
Includes Potential Outcomes (AIPW / MSM IPTW), Dynamic Panel Fixed Effects,
Convergent Cross Mapping (CCM), Interrupted Time Series (ITS),
Granger Causality, VAR IRFs, and Structural Causal DAGs.
"""

from .potential_outcomes import PotentialOutcomesEstimator
from .dynamic_panel import DynamicPanelFixedEffects
from .convergent_cross_mapping import convergent_cross_mapping
from .granger_causality import (
    check_stationarity,
    make_stationary,
    eval_granger_causality,
    eval_bidirectional_causality,
    test_granger_causality,
    test_bidirectional_causality,
    compute_pairwise_granger_matrix,
)
from .var_causal_irf import fit_causal_var_model, compute_irf_and_fevd
from .interrupted_time_series import InterruptedTimeSeriesModel
from .causal_dag import build_longitudinal_causal_dag, analyze_causal_paths

__all__ = [
    "PotentialOutcomesEstimator",
    "DynamicPanelFixedEffects",
    "convergent_cross_mapping",
    "check_stationarity",
    "make_stationary",
    "eval_granger_causality",
    "eval_bidirectional_causality",
    "test_granger_causality",
    "test_bidirectional_causality",
    "compute_pairwise_granger_matrix",
    "fit_causal_var_model",
    "compute_irf_and_fevd",
    "InterruptedTimeSeriesModel",
    "build_longitudinal_causal_dag",
    "analyze_causal_paths",
]
