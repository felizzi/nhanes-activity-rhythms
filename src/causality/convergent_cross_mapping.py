"""
Convergent Cross Mapping (CCM) Dynamical Causality Engine.
Implements Sugihara et al. (Science 2012) phase-space attractor reconstruction (Takens' Theorem)
to establish real dynamical time series causality, distinguishing true coupling from Granger spuriousness.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def delay_embed(series: np.ndarray, embedding_dim: int = 3, tau: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs delay coordinate shadow attractor manifold M_X from time series X:
      x(t) = [X(t), X(t - tau), ..., X(t - (E-1)*tau)]
    """
    n = len(series)
    max_lag = (embedding_dim - 1) * tau
    if n <= max_lag:
        raise ValueError(f"Series length {n} is too short for embedding dimension {embedding_dim} and tau {tau}.")

    vectors = []
    valid_indices = []
    for t in range(max_lag, n):
        vec = [series[t - i * tau] for i in range(embedding_dim)]
        vectors.append(vec)
        valid_indices.append(t)

    return np.array(vectors), np.array(valid_indices)


def simplex_projection(
    manifold_y: np.ndarray,
    target_x: np.ndarray,
    valid_indices: np.ndarray,
    library_size: int,
    embedding_dim: int = 3,
) -> float:
    """
    Performs nearest-neighbor simplex projection on shadow manifold M_Y to estimate X.
    Returns the Pearson correlation rho between predicted and true X.
    """
    n_points = len(manifold_y)
    lib_size = min(library_size, n_points - 1)

    # Use first lib_size points as library, remaining as evaluation (or random sub-sample)
    lib_idx = np.arange(lib_size)
    pred_idx = np.arange(n_points)

    k_neighbors = embedding_dim + 1
    dists = cdist(manifold_y[pred_idx], manifold_y[lib_idx])

    # Avoid self-matching by setting diagonal to infinity
    for i in range(min(len(pred_idx), len(lib_idx))):
        dists[i, i] = np.inf

    predicted_x = np.zeros(len(pred_idx))
    actual_x = target_x[valid_indices[pred_idx]]

    for i in range(len(pred_idx)):
        row_dists = dists[i]
        nearest_indices = np.argsort(row_dists)[:k_neighbors]
        min_dist = row_dists[nearest_indices[0]]

        if min_dist == 0:
            weights = np.zeros(k_neighbors)
            weights[0] = 1.0
        else:
            raw_weights = np.exp(-row_dists[nearest_indices] / min_dist)
            weights = raw_weights / np.sum(raw_weights)

        predicted_x[i] = np.sum(weights * target_x[valid_indices[nearest_indices]])

    # Calculate Pearson correlation rho
    valid_mask = (~np.isnan(predicted_x)) & (~np.isnan(actual_x))
    if np.sum(valid_mask) < 5 or np.std(predicted_x[valid_mask]) == 0 or np.std(actual_x[valid_mask]) == 0:
        return 0.0

    rho, _ = stats_pearson(predicted_x[valid_mask], actual_x[valid_mask])
    return float(np.clip(rho, -1.0, 1.0))


def stats_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Helper for Pearson correlation."""
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx ** 2) * np.sum(vy ** 2))
    if denom == 0:
        return 0.0, 1.0
    return float(np.sum(vx * vy) / denom), 0.0


def convergent_cross_mapping(
    series_x: pd.Series,
    series_y: pd.Series,
    embedding_dim: int = 3,
    tau: int = 1,
    lib_sizes: Optional[List[int]] = None,
) -> Dict[str, Union[List[int], List[float], bool, str]]:
    """
    Tests whether X causally forces Y using Convergent Cross Mapping (CCM).
    - If X causes Y, historical states of Y can cross-predict X: M_Y -> X.
    - Definitive proof of causality is CONVERGENCE: forecast skill rho increases with library length L.
    """
    x_vals = series_x.dropna().values
    y_vals = series_y.dropna().values
    n = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:n]
    y_vals = y_vals[:n]

    # Reconstruct shadow manifolds
    my, idx_y = delay_embed(y_vals, embedding_dim=embedding_dim, tau=tau)
    mx, idx_x = delay_embed(x_vals, embedding_dim=embedding_dim, tau=tau)

    max_lib = len(my)
    if lib_sizes is None:
        lib_sizes = [
            int(f * max_lib)
            for f in [0.15, 0.3, 0.5, 0.7, 0.95]
            if int(f * max_lib) > embedding_dim + 5
        ]
        if not lib_sizes:
            lib_sizes = [max_lib - 5]

    # Test M_Y -> X (Y predicts X implies X causes Y)
    rho_y_to_x = []
    for l_size in lib_sizes:
        rho = simplex_projection(my, x_vals, idx_y, library_size=l_size, embedding_dim=embedding_dim)
        rho_y_to_x.append(round(rho, 4))

    # Test M_X -> Y (X predicts Y implies Y causes X)
    rho_x_to_y = []
    for l_size in lib_sizes:
        rho = simplex_projection(mx, y_vals, idx_x, library_size=l_size, embedding_dim=embedding_dim)
        rho_x_to_y.append(round(rho, 4))

    # Test for convergence: rho at max library size > rho at min library size + positive skill
    is_causal_x_causes_y = bool(rho_y_to_x[-1] > rho_y_to_x[0] and rho_y_to_x[-1] > 0.25)
    is_causal_y_causes_x = bool(rho_x_to_y[-1] > rho_x_to_y[0] and rho_x_to_y[-1] > 0.25)

    if is_causal_x_causes_y and not is_causal_y_causes_x:
        direction = f"Unidirectional Dynamical Causality: {series_x.name} -> {series_y.name}"
    elif is_causal_x_causes_y and is_causal_y_causes_x:
        direction = f"Bidirectional Dynamical Coupling: {series_x.name} <-> {series_y.name}"
    elif not is_causal_x_causes_y and is_causal_y_causes_x:
        direction = f"Reverse Dynamical Causality: {series_y.name} -> {series_x.name}"
    else:
        direction = f"No Dynamical Coupling detected ({series_x.name} _|_ {series_y.name})"

    return {
        "var_x": series_x.name or "X",
        "var_y": series_y.name or "Y",
        "library_sizes": lib_sizes,
        "rho_x_causes_y": rho_y_to_x,
        "rho_y_causes_x": rho_x_to_y,
        "is_causal_x_to_y": is_causal_x_causes_y,
        "is_causal_y_to_x": is_causal_y_causes_x,
        "causal_direction": direction,
        "convergence_demonstrated": bool(rho_y_to_x[-1] > rho_y_to_x[0]),
    }
