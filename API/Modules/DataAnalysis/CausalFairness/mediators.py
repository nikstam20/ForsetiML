from .causal_graph import (
    build_causal_graph,
    preprocess_data,
    remove_highly_correlated_features,
    enforce_directionality,
)
from fairlearn.datasets import fetch_adult
from networkx import DiGraph, all_simple_paths


def identify_multi_level_mediators(sm, sensitive_attr, target_attr):
    """
    Identify multi-level mediators in the causal graph.
    Mediators are nodes on any directed path from the sensitive attribute to the target attribute.

    Args:
        sm (StructureModel): The causal graph.
        sensitive_attr (str): The sensitive attribute in the graph.
        target_attr (str): The target attribute in the graph.

    Returns:
        set: A set of mediator variables along multi-level paths.
    """
    G = DiGraph()
    for u, v, data in sm.edges(data=True):
        G.add_edge(u, v, weight=data["weight"])

    all_paths = list(all_simple_paths(G, source=sensitive_attr, target=target_attr))

    mediators = set()
    for path in all_paths:
        mediators.update(path[1:-1])
    return mediators


def main():
    data, target = fetch_adult(as_frame=True, return_X_y=True)
    data["income"] = target

    sensitive_attr = "sex"
    target_attr = "income"

    processed_data = preprocess_data(data)
    processed_data = remove_highly_correlated_features(processed_data)

    sm = build_causal_graph(processed_data, max_iter=10)

    other_statics = ["race", "age"]
    static_vars = [sensitive_attr] + other_statics

    sm = enforce_directionality(sm, static_vars, target_attr)

    mediators = identify_multi_level_mediators(sm, sensitive_attr, target_attr)


if __name__ == "__main__":
    main()
