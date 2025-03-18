from .causal_graph import preprocess_data, build_causal_graph, enforce_directionality
from networkx import DiGraph, all_simple_paths

def path_specific_fairness(
    sm, sensitive_attr, target_attr, legitimate_vars, proxy_vars
):
    """
    Assess path-specific fairness by identifying direct discrimination, indirect discrimination, and explainable bias.
    Provides explanations for each path category.

    Args:
        sm (StructureModel): The causal graph.
        sensitive_attr (str): The sensitive attribute in the graph.
        target_attr (str): The target attribute in the graph.
        legitimate_vars (list): List of legitimate explanatory variables.
        proxy_vars (list): List of proxy variables.

    Returns:
        dict: Classification of paths with explanations.
    """
    G = DiGraph()
    G.add_edges_from(sm.edges())

    all_paths = list(all_simple_paths(G, source=sensitive_attr, target=target_attr))
    results = {"direct": [], "indirect": [], "explainable": []}

    for path in all_paths:
        explanation = {"path": path, "reason": ""}
        if len(path) == 2:
            explanation["reason"] = (
                "This path directly connects the sensitive attribute to the target, "
                "indicating potential direct discrimination."
            )
            results["direct"].append(explanation)
        elif any(node in proxy_vars for node in path[1:-1]):
            explanation["reason"] = (
                "This path passes through a proxy variable, suggesting indirect discrimination."
            )
            results["indirect"].append(explanation)
        elif all(node in legitimate_vars for node in path[1:-1]):
            explanation["reason"] = (
                "This path only involves legitimate variables, representing explainable bias."
            )
            results["explainable"].append(explanation)
    return results


def no_unresolved_discrimination(sm, sensitive_attr, target_attr, resolving_vars):
    """
    Check if all paths from the sensitive attribute to the target attribute pass through at least one resolving variable.
    Provides explanations for paths that fail.

    Args:
        sm (StructureModel): The causal graph.
        sensitive_attr (str): The sensitive attribute in the graph.
        target_attr (str): The target attribute in the graph.
        resolving_vars (list): List of resolving variables.

    Returns:
        dict: Contains a boolean result and explanations for failing paths.
    """
    G = DiGraph()
    G.add_edges_from(sm.edges())

    all_paths = list(all_simple_paths(G, source=sensitive_attr, target=target_attr))
    failing_paths = []

    for path in all_paths:
        if not any(node in resolving_vars for node in path[1:-1]):
            failing_paths.append(
                {
                    "path": path,
                    "reason": (
                        "This path does not pass through any resolving variables, "
                        "indicating unresolved discrimination."
                    ),
                }
            )

    return {
        "resolved": len(failing_paths) == 0,
        "failing_paths": failing_paths,
    }


def no_proxy_discrimination(sm, sensitive_attr, target_attr, proxy_vars):
    """
    Ensure no paths from the sensitive attribute to the target attribute pass through proxy variables.
    Provides explanations for paths that fail.

    Args:
        sm (StructureModel): The causal graph.
        sensitive_attr (str): The sensitive attribute in the graph.
        target_attr (str): The target attribute in the graph.
        proxy_vars (list): List of proxy variables.

    Returns:
        dict: Contains a boolean result and explanations for failing paths.
    """
    G = DiGraph()
    G.add_edges_from(sm.edges())

    all_paths = list(all_simple_paths(G, source=sensitive_attr, target=target_attr))
    failing_paths = []

    for path in all_paths:
        if any(node in proxy_vars for node in path[1:-1]):
            failing_paths.append(
                {
                    "path": path,
                    "reason": (
                        "This path passes through a proxy variable, which violates the no proxy discrimination condition."
                    ),
                }
            )

    return {
        "no_proxies": len(failing_paths) == 0,
        "failing_paths": failing_paths,
    }


def total_effect():
    return None

def main():
    from fairlearn.datasets import fetch_adult

    data, target = fetch_adult(as_frame=True, return_X_y=True)
    data["income"] = target

    sensitive_attr = "sex"
    target_attr = "income"
    legitimate_vars = ["education", "workclass", "occupation"]
    resolving_vars = ["education"]
    proxy_vars = ["race"]

    processed_data = preprocess_data(data)
    sm = build_causal_graph(processed_data, max_iter=10)
    sm = enforce_directionality(
        sm, static_vars=[sensitive_attr, "race"], target_attr=target_attr
    )

    total_effect()

    fairness_results = path_specific_fairness(
        sm, sensitive_attr, target_attr, legitimate_vars, proxy_vars
    )

    resolved = no_unresolved_discrimination(
        sm, sensitive_attr, target_attr, resolving_vars
    )

    no_proxies = no_proxy_discrimination(sm, sensitive_attr, target_attr, proxy_vars)


if __name__ == "__main__":
    main()
