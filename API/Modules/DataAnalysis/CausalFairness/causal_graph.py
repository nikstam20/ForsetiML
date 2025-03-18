import pandas as pd
from sklearn.preprocessing import StandardScaler
from causalnex.structure.notears import from_pandas
from fairlearn.datasets import fetch_adult
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(data):
    """
    Preprocess the dataset by encoding categorical data and scaling numeric features.
    """
    categorical_data = data.apply(
        lambda col: col.astype("category") if col.dtypes == "object" else col
    )
    encoded_data = categorical_data.apply(
        lambda col: col.cat.codes if col.dtypes.name == "category" else col
    )

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(encoded_data), columns=encoded_data.columns
    )
    return scaled_data


def remove_highly_correlated_features(data, threshold=0.95):
    """
    Remove features with a correlation higher than the specified threshold.
    """
    corr_matrix = data.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    return data.drop(columns=to_drop)


def build_causal_graph(data, max_iter=10, w_threshold=0.1):
    """
    Build a causal graph automatically using the NOTEARS algorithm.
    """
    sm = from_pandas(data, max_iter=max_iter, w_threshold=w_threshold)

    return sm


def enforce_directionality(sm, static_vars, target_attr):
    """
    Enforce directionality in the causal graph:
    - Static variables should only have outgoing edges.
    - Target attribute should only have incoming edges.
    - Non-static, non-target variables are left to the learned causal structure.

    Args:
        sm (StructureModel): The causal graph.
        static_vars (list): List of static variables with only outgoing edges.
        target_attr (str): The target attribute with only incoming edges.

    Returns:
        StructureModel: The modified causal graph.
    """
    modified_sm = sm.copy()

    for static_var in static_vars:
        for neighbor in list(modified_sm.predecessors(static_var)):
            try:
                weight = modified_sm.get_edge_data(neighbor, static_var)["weight"]
                modified_sm.remove_edge(neighbor, static_var)
                modified_sm.add_edge(static_var, neighbor, weight=weight)
            except Exception as e:
                print(
                    f"Error processing static_var: {static_var}, neighbor: {neighbor}"
                )
                print(f"Error details: {e}")
                raise

    for neighbor in list(modified_sm.successors(target_attr)):
        try:
            weight = modified_sm.get_edge_data(target_attr, neighbor)["weight"]
            modified_sm.remove_edge(target_attr, neighbor)
            modified_sm.add_edge(neighbor, target_attr, weight=weight)
        except Exception as e:
            print(f"Error processing target_attr: {target_attr}, neighbor: {neighbor}")
            print(f"Error details: {e}")
            raise

    return modified_sm


def export_causal_graph(sm, sensitive_attr, target_attr):
    """
    Export the causal graph as a JSON-compatible dictionary, including edge weights.
    """
    edges_with_weights = [
        {"source": u, "target": v, "weight": w} for u, v, w in sm.edges(data="weight")
    ]
    return {
        "nodes": [{"id": node} for node in sm.nodes()],
        "edges": edges_with_weights,
        "sensitive_attr": sensitive_attr,
        "target_attr": target_attr,
    }


def visualize_graph(sm):
    """
    Visualize the causal graph using networkx and matplotlib.
    """
    G = nx.DiGraph()

    G.add_edges_from(sm.edges(data=True))

    pos = nx.spring_layout(G)
    edge_weights = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=700)
    nx.draw_networkx_labels(G, pos)

    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_weights.items()}
    )

    plt.title("Causal Graph")
    plt.show()


def main():
    data, target = fetch_adult(as_frame=True, return_X_y=True)
    data["income"] = target

    sensitive_attr = "sex"
    other_statics = ["race", "age"]
    static_vars = [sensitive_attr] + other_statics
    target_attr = "income"

    processed_data = preprocess_data(data)

    sm = build_causal_graph(processed_data, max_iter=10)

    sm = enforce_directionality(sm, static_vars, target_attr)

    graph_json = export_causal_graph(sm, sensitive_attr, target_attr)

    visualize_graph(sm)


if __name__ == "__main__":
    main()
