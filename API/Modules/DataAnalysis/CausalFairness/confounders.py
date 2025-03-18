from .causal_graph import (
    build_causal_graph,
    preprocess_data,
    remove_highly_correlated_features,
    enforce_directionality,
)
from fairlearn.datasets import fetch_adult
from networkx import DiGraph


def identify_confounders(sm, sensitive_attr, target_attr):
    """
    Identify confounders in the causal graph.
    Confounders are nodes with edges pointing to both the sensitive attribute and the target attribute.

    Args:
        sm (StructureModel): The causal graph.
        sensitive_attr (str): The sensitive attribute in the graph.
        target_attr (str): The target attribute in the graph.

    Returns:
        set: A set of confounder variables.
    """
    G = DiGraph()
    for u, v, data in sm.edges(data=True):
        G.add_edge(u, v, weight=data["weight"])

    confounders = set()
    for node in G.nodes():
        if G.has_edge(node, sensitive_attr) and G.has_edge(node, target_attr):
            confounders.add(node)

    return confounders


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

    confounders = identify_confounders(sm, sensitive_attr, target_attr)

if __name__ == "__main__":
    main()
