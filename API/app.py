from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from DataAnalysis.CausalFairnessRouter import causal_fairness_bp
import pandas as pd
from DataAnalysis.GroupFairnessRouter import group_metrics_bp
from Model.model_training_inference import model_bp
from BiasMitigation.bias_mitigation import bias_mitigation_bp
from IndividualFairness.counterfacutals import counterfactual_bp
from KnowledgeGraph.graph import knowledge_graph_bp


app = Flask(__name__)

from flask_cors import CORS

CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(causal_fairness_bp, url_prefix="/api/causal")
app.register_blueprint(group_metrics_bp, url_prefix="/api/group")
app.register_blueprint(model_bp, url_prefix="/api/model")
app.register_blueprint(bias_mitigation_bp, url_prefix="/api/debias")
app.register_blueprint(counterfactual_bp, url_prefix="/api/counterfactual")
app.register_blueprint(knowledge_graph_bp, url_prefix="/api/knowledge-graph")


@app.route("/")
def root():
    return {"message": "Welcome to the Fairness API with Flask!"}


UPLOAD_FOLDER = "./uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        print("Error: 'file' not in request.files")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        print("Error: File name is empty")
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)

    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        print(
            f"Upload folder {app.config['UPLOAD_FOLDER']} does not exist. Creating it..."
        )
        os.makedirs(app.config["UPLOAD_FOLDER"])

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(filepath)
    print(f"File saved successfully at {filepath}")

    return jsonify({"message": "File uploaded successfully", "path": filepath}), 200


@app.route("/api/preview-dataset", methods=["GET"])
def preview_dataset():
    """
    Endpoint to preview the dataset.
    """
    file_path = request.args.get("path")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 400

    try:
        df = pd.read_csv(file_path)

        return (
            jsonify(
                {
                    "columns": df.columns.tolist(),
                    "preview": df.head()
                    .replace({pd.NA: None, float("nan"): None})
                    .to_dict(orient="records"),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
