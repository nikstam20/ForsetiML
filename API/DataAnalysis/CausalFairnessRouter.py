from flask import Blueprint, request, jsonify
from Modules.DataAnalysis.CausalFairness.causal_graph import (
    build_causal_graph,
    preprocess_data,
    enforce_directionality,
    export_causal_graph,
)
from Modules.DataAnalysis.CausalFairness.causal_metrics import (
    total_effect,
    path_specific_fairness,
    no_unresolved_discrimination,
    no_proxy_discrimination,
)
from Modules.DataAnalysis.CausalFairness.confounders import identify_confounders
from Modules.DataAnalysis.CausalFairness.mediators import identify_multi_level_mediators
from networkx import DiGraph
from causalnex.structure import StructureModel

causal_fairness_bp = Blueprint("causal_fairness", __name__)

from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)


@causal_fairness_bp.route("/create_causal_graph", methods=["POST"])
def create_causal_graph():
    """
    Endpoint to create and process a causal graph.
    """
    current_path = os.getcwd()

    data_path = request.json.get("data_path")

    sensitive_attr = request.json.get("sensitive_attr")
    target_attr = request.json.get("target_attr")
    static_vars = request.json.get("static_vars")
    if not data_path or not sensitive_attr or not target_attr:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        data = pd.read_csv(data_path)
        processed_data = preprocess_data(data)
        sm = build_causal_graph(processed_data, max_iter=10)
        sm = enforce_directionality(sm, static_vars, target_attr)
        graph_json = export_causal_graph(sm, sensitive_attr, target_attr)

        return (
            jsonify(
                {"message": "Causal graph created successfully", "graph": graph_json}
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/confounders", methods=["POST"])
def confounders():
    try:
        data = request.json

        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        nodes = data.get("nodes")
        edges = data.get("edges")

        G = DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

        confounders_set = identify_confounders(G, sensitive_attr, target_attr)
        return jsonify({"confounders": list(confounders_set)})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/mediators", methods=["POST"])
def mediators():
    """
    Endpoint to identify mediators.
    """
    try:
        data = request.json

        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not sensitive_attr or not target_attr:
            return (
                jsonify({"error": "Sensitive and target attributes are required"}),
                400,
            )

        G = DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        mediators = identify_multi_level_mediators(G, sensitive_attr, target_attr)

        response_data = {
            "mediators": list(mediators),
        }

        return jsonify(response_data)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/path_specific_fairness", methods=["POST"])
def path_specific_fairness_api():
    try:
        data = request.json
        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        legitimate_vars = data.get("legitimate_vars", [])
        proxy_vars = data.get("proxy_vars", [])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not sensitive_attr or not target_attr:
            return (
                jsonify({"error": "Sensitive and target attributes are required"}),
                400,
            )

        G = DiGraph()
        G.add_nodes_from([node["id"] for node in nodes])
        G.add_edges_from([(edge["source"], edge["target"]) for edge in edges])

        results = path_specific_fairness(
            G, sensitive_attr, target_attr, legitimate_vars, proxy_vars
        )
        return jsonify(results)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/no_unresolved_discrimination", methods=["POST"])
def no_unresolved_discrimination_api():
    try:
        data = request.json
        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        resolving_vars = data.get("resolving_vars", [])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not sensitive_attr or not target_attr:
            return (
                jsonify({"error": "Sensitive and target attributes are required"}),
                400,
            )

        G = DiGraph()
        G.add_nodes_from([node["id"] for node in nodes])
        G.add_edges_from([(edge["source"], edge["target"]) for edge in edges])

        resolved = no_unresolved_discrimination(
            G, sensitive_attr, target_attr, resolving_vars
        )
        return jsonify({"resolved": resolved})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/no_proxy_discrimination", methods=["POST"])
def no_proxy_discrimination_api():
    try:
        data = request.json
        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        proxy_vars = data.get("proxy_vars", [])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not sensitive_attr or not target_attr:
            return (
                jsonify({"error": "Sensitive and target attributes are required"}),
                400,
            )

        G = DiGraph()
        G.add_nodes_from([node["id"] for node in nodes])
        G.add_edges_from([(edge["source"], edge["target"]) for edge in edges])

        no_proxies = no_proxy_discrimination(G, sensitive_attr, target_attr, proxy_vars)
        return jsonify({"no_proxies": no_proxies})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@causal_fairness_bp.route("/handle_causal_elements", methods=["POST"])
def handle_causal_elements():
    """
    Endpoint to handle mediators and confounders, and return the updated causal graph.
    """
    try:
        data = request.json
        sensitive_attr = data.get("sensitive_attr")
        target_attr = data.get("target_attr")
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not sensitive_attr or not target_attr:
            return (
                jsonify({"error": "Sensitive and target attributes are required"}),
                400,
            )

        G = DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        mediators = identify_multi_level_mediators(G, sensitive_attr, target_attr)

        confounders = identify_confounders(G, sensitive_attr, target_attr)

        graph_json = {
            "nodes": [{"id": node} for node in G.nodes()],
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "weight": G.edges[edge]["weight"],
                }
                for edge in G.edges()
            ],
        }

        response_data = {
            "message": "Causal elements processed successfully",
            "graph": graph_json,
            "mediators": list(mediators),
            "confounders": list(confounders),
        }

        return jsonify(response_data), 200

    except Exception as e:
        print("Error in handling causal elements:", str(e))
        return jsonify({"error": str(e)}), 500
