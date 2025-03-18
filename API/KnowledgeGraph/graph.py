from neo4j import GraphDatabase
from flask import Blueprint, request, jsonify

knowledge_graph_bp = Blueprint("graph", __name__)

NEO4J_URI = "bolt://localhost:7687"  
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_methods_without_limitations_and_conflicts(limitations, conflicts):
    query = """
    MATCH (m:Method)
    WHERE NOT EXISTS { MATCH (m)-[:HAS_LIMITATION]->(l:Limitation) WHERE l.name IN $limitations }
    AND NOT EXISTS { MATCH (m)-[:CAUSES]->(c:Conflict) WHERE c.name IN $conflicts }
    RETURN m.name AS name, m.description AS description
    """
    with driver.session() as session:
        result = session.run(query, limitations=limitations, conflicts=conflicts)
        return [{"name": record["name"], "description": record["description"]} for record in result]

def get_metrics_without_limitations_and_in_fairness_notions(limitations, fairness_notions):
    query = """
    MATCH (metric:Metric)-[:CONFORMS_TO]->(notion:FairnessNotion)
    WHERE notion.name IN $fairness_notions
    AND NOT EXISTS { MATCH (metric)-[:HAS_LIMITATION]->(l:Limitation) WHERE l.name IN $limitations }
    RETURN metric.name AS name, metric.formula AS formula, metric.category AS category
    """
    with driver.session() as session:
        result = session.run(query, limitations=limitations, fairness_notions=fairness_notions)
        return [{"name": record["name"], "formula": record["formula"], "category": record["category"]} for record in result]

@knowledge_graph_bp.route("/query", methods=["POST"])
def query_graph():
    data = request.json
    limitations = data.get("limitations", [])
    conflicts = data.get("conflicts", [])
    fairness_notions = data.get("fairness_notions", [])
    methods = get_methods_without_limitations_and_conflicts(limitations, conflicts)
    metrics = get_metrics_without_limitations_and_in_fairness_notions(limitations, fairness_notions)

    return jsonify({"methods": methods, "metrics": metrics})