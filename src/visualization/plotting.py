"""
Publication-Quality Visualization Suite for NHANES Causal Time Series Analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import config

# Set elegant academic aesthetic
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True,
    }
)


def plot_longitudinal_cohort_trends(
    macro_df: pd.DataFrame,
    pseudo_df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plots multi-cycle national macro trends alongside Deaton pseudo-panel cohort trajectories.
    """
    out_file = output_path or (config.FIGURES_DIR / "1_longitudinal_cohort_trends.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    metrics = [
        ("daily_sugar_g", "Dietary Sugar Intake (g/day)", "#E63946"),
        ("bmi", "Body Mass Index (BMI, kg/m²)", "#457B9D"),
        ("hba1c", "Glycated Hemoglobin (HbA1c, %)", "#2A9D8F"),
        ("systolic_bp", "Systolic Blood Pressure (mmHg)", "#E76F51"),
    ]

    for ax, (var, label, color) in zip(axes.flatten(), metrics):
        # Plot cohort trajectories if available
        if pseudo_df is not None and "cohort" in pseudo_df.columns:
            for cohort_name, grp in pseudo_df.groupby("cohort"):
                ax.plot(
                    grp["survey_year"],
                    grp[var],
                    linestyle="--",
                    alpha=0.35,
                    linewidth=1.5,
                    label=f"{cohort_name}",
                )

        # Plot weighted national population macro time series
        ax.plot(
            macro_df["survey_year"],
            macro_df[var],
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=7,
            label="National Population Mean",
        )

        ax.set_title(label, fontweight="bold", pad=8)
        ax.set_ylabel(label.split("(")[-1].replace(")", "").strip())
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[1, 0].set_xlabel("Survey Cycle Year")
    axes[1, 1].set_xlabel("Survey Cycle Year")
    axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "NHANES Longitudinal Trajectories (1999–2018): Population Mean & Deaton Pseudo-Cohorts",
        fontweight="bold",
        y=1.02,
    )
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_granger_causality_matrix(
    p_matrix: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generates a heatmap of pairwise Granger causality p-values with statistical significance stars.
    """
    out_file = output_path or (config.FIGURES_DIR / "2_granger_causality_matrix.png")
    fig, ax = plt.subplots(figsize=(9, 7))

    # Mask diagonal
    mask = np.eye(len(p_matrix), dtype=bool)

    # Format annotations
    annot = np.empty(p_matrix.shape, dtype=object)
    for i in range(len(p_matrix)):
        for j in range(len(p_matrix)):
            if i == j:
                annot[i, j] = "-"
            else:
                p = p_matrix.iloc[i, j]
                if np.isnan(p):
                    annot[i, j] = "N/A"
                elif p < 0.01:
                    annot[i, j] = f"{p:.3f}**"
                elif p < 0.05:
                    annot[i, j] = f"{p:.3f}*"
                else:
                    annot[i, j] = f"{p:.3f}"

    cmap = sns.diverging_palette(145, 10, s=85, l=55, as_cmap=True)

    sns.heatmap(
        p_matrix,
        mask=mask,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=0.20,
        cbar_kws={"label": "Granger Causality p-value"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_title("Pairwise Granger Causality p-Value Matrix\n(Row Granger-causes Column; *p<0.05, **p<0.01)", fontweight="bold", pad=12)
    ax.set_xlabel("Effect Variable (Y)", fontweight="bold")
    ax.set_ylabel("Cause Variable (X)", fontweight="bold")

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_var_impulse_responses(
    var_results: object,
    cause_var: str = "daily_sugar_g",
    effect_vars: Optional[List[str]] = None,
    periods: int = 8,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plots dynamic orthogonalized impulse response functions (OIRF) showing how an exogenous shock
    in cause_var impacts downstream metabolic biomarkers over future horizons.
    """
    out_file = output_path or (config.FIGURES_DIR / "3_var_impulse_responses.png")
    irf = var_results.irf(periods=periods)
    names = var_results.names

    if cause_var not in names:
        cause_var = names[0]
    cause_idx = names.index(cause_var)

    target_vars = effect_vars or [v for v in names if v != cause_var]
    n_targets = len(target_vars)

    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5), sharex=True)
    if n_targets == 1:
        axes = [axes]

    horizons = np.arange(periods + 1)

    for ax, eff_var in zip(axes, target_vars):
        eff_idx = names.index(eff_var)

        response = irf.orth_irfs[:, eff_idx, cause_idx]
        stderr = irf.stderr(orth=True)[:, eff_idx, cause_idx]

        ci_lower = response - 1.96 * stderr
        ci_upper = response + 1.96 * stderr

        ax.plot(horizons, response, color="#1D3557", linewidth=2.5, marker="o", markersize=5, label="OIRF Response")
        ax.fill_between(horizons, ci_lower, ci_upper, color="#457B9D", alpha=0.25, label="95% Confidence Band")
        ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2, alpha=0.8)

        ax.set_title(f"Shock in {cause_var}\n→ Response of {eff_var}", fontweight="bold", fontsize=11)
        ax.set_xlabel("Forecast Horizon (Cycles)")
        ax.set_ylabel(f"Δ {eff_var}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"VAR Orthogonalized Causal Impulse Response Functions (Shock: 1 SD in {cause_var})", fontweight="bold", y=1.04)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_interrupted_time_series(
    its_model: object,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plots the observed series, factual model fit, and projected counterfactual trajectory
    for Interrupted Time Series quasi-experimental causal analysis.
    """
    out_file = output_path or (config.FIGURES_DIR / "4_interrupted_time_series.png")
    df_proj = its_model.get_projection_dataframe()
    var_name = its_model.outcome_var

    fig, ax = plt.subplots(figsize=(10, 6))

    # Observed points
    ax.scatter(df_proj["T_time"], df_proj[var_name], color="#1D3557", s=70, zorder=5, label=f"Observed {var_name}")

    # Factual model fit
    ax.plot(df_proj["T_time"], df_proj["y_factual_pred"], color="#2A9D8F", linewidth=2.8, label="Factual Policy Path")

    # Counterfactual post-intervention path
    post_mask = df_proj["D_intervention"] == 1
    post_times = df_proj.loc[post_mask, "T_time"]
    counterfactual_vals = df_proj.loc[post_mask, "y_counterfactual"]

    ax.plot(
        post_times,
        counterfactual_vals,
        color="#E63946",
        linestyle="--",
        linewidth=2.5,
        label="Counterfactual (Without Policy)",
    )

    # Shaded causal effect region
    factual_post = df_proj.loc[post_mask, "y_factual_pred"]
    ax.fill_between(
        post_times,
        counterfactual_vals,
        factual_post,
        color="#E63946",
        alpha=0.18,
        label="Causal Treatment Effect (ATE)",
    )

    # Vertical line at intervention cycle
    interv_t = df_proj.loc[df_proj["D_intervention"] == 1, "T_time"].min()
    ax.axvline(interv_t, color="#6C757D", linestyle=":", linewidth=2, label="Intervention Enactment")

    ax.set_title(f"Quasi-Experimental Causal Impact: Interrupted Time Series for {var_name}", fontweight="bold", pad=12)
    ax.set_xlabel("Time Index (2-Year Survey Cycles)", fontweight="bold")
    ax.set_ylabel(var_name, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_causal_dag(
    dag: nx.DiGraph,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Visualizes the time-unrolled structural causal Directed Acyclic Graph (DAG).
    """
    out_file = output_path or (config.FIGURES_DIR / "5_causal_dag.png")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pre-calculated layered positions for clean display
    pos = {
        "Demographics_(Age,Sex,PIR)": (0.0, 2.8),
        "Public_Health_Policy_(t)": (3.5, 2.8),
        "Dietary_Sugar_(t-1)": (0.5, 1.8),
        "Sedentary_Time_(t-1)": (0.5, 0.4),
        "BMI_(t-1)": (1.8, 1.1),
        "HbA1c_(t-1)": (2.2, 0.0),
        "Dietary_Sugar_(t)": (3.5, 1.8),
        "Sedentary_Time_(t)": (3.5, 0.4),
        "BMI_(t)": (4.8, 1.1),
        "HbA1c_(t)": (5.2, 0.0),
    }

    color_map = {
        "exposure": "#F4A261",
        "mediator": "#2A9D8F",
        "outcome": "#E76F51",
        "confounder": "#A8DADC",
        "policy_intervention": "#E9C46A",
    }

    node_colors = [color_map.get(dag.nodes[n].get("type", "exposure"), "#CBD5E1") for n in dag.nodes()]

    nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=3200, edgecolors="#2B2D42", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(dag, pos, font_size=8, font_weight="bold", ax=ax)

    # Draw straight and curved arrows
    nx.draw_networkx_edges(
        dag,
        pos,
        edge_color="#4A5568",
        arrowsize=18,
        arrowstyle="-|>",
        node_size=3200,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
        width=1.5,
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="Confounder", markerfacecolor="#A8DADC", markersize=12),
        plt.Line2D([0], [0], marker="o", color="w", label="Exposure (Diet/Activity)", markerfacecolor="#F4A261", markersize=12),
        plt.Line2D([0], [0], marker="o", color="w", label="Mediator (BMI)", markerfacecolor="#2A9D8F", markersize=12),
        plt.Line2D([0], [0], marker="o", color="w", label="Biomarker Outcome (HbA1c)", markerfacecolor="#E76F51", markersize=12),
        plt.Line2D([0], [0], marker="o", color="w", label="Policy Intervention", markerfacecolor="#E9C46A", markersize=12),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title("Longitudinal Structural Causal Model (Time-Unrolled DAG)", fontweight="bold", pad=12)
    ax.axis("off")

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_accelerometer_causality(
    accel_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plots high-frequency 7-day accelerometer time series showing dynamic coupling
    between physical activity intensity and delayed metabolic expenditure / heart rate.
    """
    out_file = output_path or (config.FIGURES_DIR / "6_accelerometer_dynamics.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Take first 72 hours for high visual clarity
    subset = accel_df.iloc[:72]

    # Physical Activity Counts
    ax1.plot(subset["timestamp"], subset["activity_counts"], color="#2B2D42", linewidth=1.5, label="Activity Counts (CPM)")
    ax1.set_ylabel("Activity Intensity (counts/min)", fontweight="bold")
    ax1.set_title("NHANES High-Frequency Wearable Accelerometer: Dynamic Activity vs. Metabolic Dynamics", fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Metabolic rate and Heart rate
    ax2.plot(subset["timestamp"], subset["metabolic_rate_mets"], color="#E63946", linewidth=2.0, label="Metabolic Rate (METs)")
    ax2.set_ylabel("METs", color="#E63946", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#E63946")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(subset["timestamp"], subset["heart_rate_bpm"], color="#457B9D", linewidth=1.8, linestyle="--", label="Heart Rate (bpm)")
    ax2_twin.set_ylabel("Heart Rate (bpm)", color="#457B9D", fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor="#457B9D")
    ax2_twin.grid(False)

    ax2.set_xlabel("Time (Hourly Intervals across 72 Hours)", fontweight="bold")

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_covariate_balance_love_plot(
    balance_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Renders an epidemiological Love Plot showing Standardized Mean Differences (SMD)
    before and after Inverse Probability Treatment Weighting (IPTW).
    Demonstrates successful elimination of baseline confounding across confounders.
    """
    out_file = output_path or (config.FIGURES_DIR / "7_covariate_balance_love_plot.png")
    fig, ax = plt.subplots(figsize=(9, 6))

    y_pos = np.arange(len(balance_df))

    # Plot unweighted vs weighted SMDs
    ax.scatter(balance_df["smd_unweighted"], y_pos, color="#E63946", s=80, marker="o", label="Unadjusted (Confounded)", zorder=4)
    ax.scatter(balance_df["smd_weighted"], y_pos, color="#2A9D8F", s=100, marker="D", label="Weighted (IPTW Balanced)", zorder=5)

    for i in range(len(balance_df)):
        ax.plot(
            [balance_df["smd_unweighted"].iloc[i], balance_df["smd_weighted"].iloc[i]],
            [y_pos[i], y_pos[i]],
            color="#A8DADC",
            linestyle=":",
            linewidth=1.8,
            zorder=3,
        )

    # 0.10 threshold line for acceptable balance
    ax.axvline(0.10, color="#E76F51", linestyle="--", linewidth=1.5, label="Balance Threshold (SMD = 0.10)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df["confounder"], fontweight="bold")
    ax.set_xlabel("Absolute Standardized Mean Difference (SMD)", fontweight="bold")
    ax.set_title("Potential Outcomes: Covariate Balance Love Plot\n(Before vs. After Inverse Probability Weighting)", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_ccm_convergence(
    ccm_result: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plots Sugihara's Convergent Cross Mapping (CCM) convergence curves:
    Forecast skill rho vs library length L. Monotonic convergence confirms true dynamical causality.
    """
    out_file = output_path or (config.FIGURES_DIR / "8_ccm_dynamical_causality.png")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    lib_sizes = ccm_result["library_sizes"]
    rho_xy = ccm_result["rho_x_causes_y"]
    rho_yx = ccm_result["rho_y_causes_x"]
    vx = ccm_result["var_x"]
    vy = ccm_result["var_y"]

    ax.plot(
        lib_sizes,
        rho_xy,
        color="#2A9D8F",
        linewidth=2.8,
        marker="o",
        markersize=7,
        label=f"M_{vy} -> {vx} (Testing {vx} causes {vy})",
    )
    ax.plot(
        lib_sizes,
        rho_yx,
        color="#E63946",
        linewidth=2.2,
        marker="s",
        markersize=6,
        linestyle="--",
        label=f"M_{vx} -> {vy} (Testing {vy} causes {vx})",
    )

    ax.set_title(f"Convergent Cross Mapping (CCM) Dynamical Causality\n{ccm_result['causal_direction']}", fontweight="bold", pad=12)
    ax.set_xlabel("Library Length (L observations)", fontweight="bold")
    ax.set_ylabel("Cross-Map Forecast Skill (Pearson ρ)", fontweight="bold")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file

