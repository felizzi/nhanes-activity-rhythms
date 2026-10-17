"""
Structural Causal Model and Longitudinal Directed Acyclic Graph (DAG).
Formulates time-unrolled causal diagrams, backdoor criterion paths, and confounding structures.
"""

from typing import Dict, List, Set, Tuple, Union
import networkx as nx


def build_longitudinal_causal_dag() -> nx.DiGraph:
    """
    Constructs a time-unrolled Directed Acyclic Graph (DAG) for longitudinal NHANES analysis.
    Time stages:
      - t-1: Lagged exposures, mediators, and outcomes
      - t: Contemporaneous exposures, mediators, and outcomes
      - Z: Invariant / baseline demographic confounders (Age, Sex, Race)
    """
    dag = nx.DiGraph()

    # Define nodes with domain classifications
    nodes = {
        # Time t-1
        "Dietary_Sugar_(t-1)": {"time": "t-1", "type": "exposure"},
        "Sedentary_Time_(t-1)": {"time": "t-1", "type": "exposure"},
        "BMI_(t-1)": {"time": "t-1", "type": "mediator"},
        "HbA1c_(t-1)": {"time": "t-1", "type": "outcome"},
        # Time t
        "Dietary_Sugar_(t)": {"time": "t", "type": "exposure"},
        "Sedentary_Time_(t)": {"time": "t", "type": "exposure"},
        "BMI_(t)": {"time": "t", "type": "mediator"},
        "HbA1c_(t)": {"time": "t", "type": "outcome"},
        # Confounders / Exogenous
        "Demographics_(Age,Sex,PIR)": {"time": "invariant", "type": "confounder"},
        "Public_Health_Policy_(t)": {"time": "t", "type": "policy_intervention"},
    }

    for node, attrs in nodes.items():
        dag.add_node(node, **attrs)

    # Define causal edges (arrows represent causal influence)
    edges = [
        # Autoregressive persistence over time
        ("Dietary_Sugar_(t-1)", "Dietary_Sugar_(t)"),
        ("Sedentary_Time_(t-1)", "Sedentary_Time_(t)"),
        ("BMI_(t-1)", "BMI_(t)"),
        ("HbA1c_(t-1)", "HbA1c_(t)"),
        # Causal forward pathways
        ("Dietary_Sugar_(t-1)", "BMI_(t)"),
        ("Sedentary_Time_(t-1)", "BMI_(t)"),
        ("Dietary_Sugar_(t-1)", "HbA1c_(t)"),
        ("BMI_(t-1)", "HbA1c_(t)"),
        ("BMI_(t)", "HbA1c_(t)"),
        # Behavioral feedback (Reverse Causality): High past HbA1c leads to diet intervention
        ("HbA1c_(t-1)", "Dietary_Sugar_(t)"),
        # Confounding from Demographics
        ("Demographics_(Age,Sex,PIR)", "Dietary_Sugar_(t-1)"),
        ("Demographics_(Age,Sex,PIR)", "Dietary_Sugar_(t)"),
        ("Demographics_(Age,Sex,PIR)", "Sedentary_Time_(t-1)"),
        ("Demographics_(Age,Sex,PIR)", "Sedentary_Time_(t)"),
        ("Demographics_(Age,Sex,PIR)", "BMI_(t-1)"),
        ("Demographics_(Age,Sex,PIR)", "BMI_(t)"),
        ("Demographics_(Age,Sex,PIR)", "HbA1c_(t-1)"),
        ("Demographics_(Age,Sex,PIR)", "HbA1c_(t)"),
        # Policy Intervention
        ("Public_Health_Policy_(t)", "Dietary_Sugar_(t)"),
        ("Public_Health_Policy_(t)", "HbA1c_(t)"),
    ]

    dag.add_edges_from(edges)
    return dag


def analyze_causal_paths(
    dag: nx.DiGraph,
    source: str = "Dietary_Sugar_(t-1)",
    target: str = "HbA1c_(t)",
) -> Dict[str, Union[List[List[str]], List[str]]]:
    """
    Identifies all directed causal pathways and mediators between exposure and outcome.
    """
    all_paths = list(nx.all_simple_paths(dag, source=source, target=target))
    direct_paths = [p for p in all_paths if len(p) == 2]
    mediated_paths = [p for p in all_paths if len(p) > 2]

    return {
        "source": source,
        "target": target,
        "all_paths": all_paths,
        "direct_path": direct_paths,
        "mediated_paths": mediated_paths,
        "mediators": sorted(list(set(n for p in mediated_paths for n in p[1:-1]))),
    }
