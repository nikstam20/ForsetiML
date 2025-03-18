from flask import Blueprint, request, jsonify
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

counterfactual_bp = Blueprint("counterfactual", __name__)

MODEL_FILE_CLASSIFICATION = "xgboost_classification_model.pkl"
LABEL_ENCODERS_FILE = "label_encoders.pkl"
TARGET_LABEL_ENCODER_FILE = "target_label_encoder.pkl"


def load_model():
    if not os.path.exists(MODEL_FILE_CLASSIFICATION):
        raise FileNotFoundError(f"Model file '{MODEL_FILE_CLASSIFICATION}' not found.")
    with open(MODEL_FILE_CLASSIFICATION, "rb") as f:
        return pickle.load(f)


def load_encoders():
    if not os.path.exists(LABEL_ENCODERS_FILE):
        raise FileNotFoundError("Label encoders file not found.")
    with open(LABEL_ENCODERS_FILE, "rb") as f:
        return pickle.load(f)


def encode_categorical_columns(df, encoders):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df


@counterfactual_bp.route("/find_similar", methods=["POST"])
def find_similar_entities():
    """
    Find similar entities and return possible values for the sensitive attribute.
    """
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        selected_entity_index = request.json.get("selected_entity_index")
        num_neighbors = request.json.get("num_neighbors", 5)
        sensitive_attribute = request.json.get("sensitive_attribute")

        if (
            not data_path
            or target_column is None
            or selected_entity_index is None
            or not sensitive_attribute
        ):
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at '{data_path}'"}), 400

        df = pd.read_csv(data_path)

        if target_column not in df.columns or sensitive_attribute not in df.columns:
            return (
                jsonify(
                    {"error": "Target or sensitive attribute not found in dataset"}
                ),
                400,
            )

        if selected_entity_index >= len(df):
            return jsonify({"error": "Selected entity index is out of bounds"}), 400

        possible_sensitive_values = df[sensitive_attribute].unique().tolist()

        encoders = load_encoders()
        df_encoded = encode_categorical_columns(df.copy(), encoders)
        X = df_encoded.drop(columns=[target_column])

        nn_model = NearestNeighbors(
            n_neighbors=min(num_neighbors, len(df)), metric="euclidean"
        )
        nn_model.fit(X)
        sample = X.iloc[selected_entity_index : selected_entity_index + 1]
        distances, indices = nn_model.kneighbors(sample)
        neighbors = df.iloc[indices[0]]

        similar_entities = []
        for idx, row in neighbors.iterrows():
            similar_entities.append(
                {
                    "index": idx,
                    "features": row.to_dict(),
                    "prediction": int(load_model().predict([X.iloc[idx]])[0]),
                }
            )

        return (
            jsonify(
                {
                    "similar_entities": similar_entities,
                    "possible_sensitive_values": possible_sensitive_values,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from collections import defaultdict
import json

counterfactual_bp = Blueprint("counterfactual", __name__)

MODEL_FILE_CLASSIFICATION = "xgboost_classification_model.pkl"
LABEL_ENCODERS_FILE = "label_encoders.pkl"


def load_model():
    if not os.path.exists(MODEL_FILE_CLASSIFICATION):
        raise FileNotFoundError(f"Model file '{MODEL_FILE_CLASSIFICATION}' not found.")
    with open(MODEL_FILE_CLASSIFICATION, "rb") as f:
        return pickle.load(f)


def load_encoders():
    if not os.path.exists(LABEL_ENCODERS_FILE):
        raise FileNotFoundError("Label encoders file not found.")
    with open(LABEL_ENCODERS_FILE, "rb") as f:
        return pickle.load(f)


def encode_categorical_columns(df, encoders):
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df


from flask import jsonify, make_response
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np


@counterfactual_bp.route("/cluster_entities", methods=["POST"])
def cluster_entities():
    """
    Cluster the dataset using t-SNE + KMeans to find distinct, well-separated clusters,
    and include predictions for each entity while retaining their original indices.
    """
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        sensitive_attribute = request.json.get("sensitive_attribute")
        max_entities_per_cluster = request.json.get("max_entities_per_cluster", 30)
        num_clusters = request.json.get("num_clusters", 5)

        if not data_path or target_column is None or not sensitive_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at '{data_path}'"}), 400

        df = pd.read_csv(data_path)

        if target_column not in df.columns or sensitive_attribute not in df.columns:
            return (
                jsonify(
                    {"error": "Target or sensitive attribute not found in dataset"}
                ),
                400,
            )

        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42).reset_index(drop=False)
            df.rename(columns={"index": "original_index"}, inplace=True)
        else:
            df = df.reset_index(drop=False)
            df.rename(columns={"index": "original_index"}, inplace=True)

        # df_encoded = df.drop(columns=[sensitive_attribute])
        df_encoded = df.copy()
        encoders = load_encoders()
        df_encoded = encode_categorical_columns(df_encoded, encoders)
        X = df_encoded.drop(columns=[target_column, "original_index"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
        X_tsne = tsne.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_tsne)

        model = load_model()

        predictions = model.predict(X)

        grouped_clusters = defaultdict(list)
        for i in range(len(df)):
            cluster_id = cluster_labels[i]
            entity = {
                "index": int(df.loc[i, "original_index"]) + 1,  # Use original index
                "coordinates": [float(X_tsne[i, 0]), float(X_tsne[i, 1])],
                "features": df.drop(columns=["original_index"])
                .iloc[i]
                .replace({np.nan: None})
                .to_dict(),
                "prediction": float(predictions[i]),  # Include prediction
            }
            grouped_clusters[cluster_id].append(entity)

        final_clusters = {
            str(cluster_id): entities[:max_entities_per_cluster]
            for cluster_id, entities in grouped_clusters.items()
        }

        return jsonify(
            {
                "clusters": final_clusters,
                "possible_sensitive_values": df[sensitive_attribute].unique().tolist(),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@counterfactual_bp.route("/generate", methods=["POST"])
def generate_counterfactuals_for_entity():
    """
    Generate counterfactuals by changing the sensitive attribute to a user-selected value.
    """
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        selected_entity_index = request.json.get("selected_entity_index") - 1
        print(selected_entity_index)
        sensitive_attribute = request.json.get("sensitive_attribute")
        new_sensitive_value = request.json.get("new_sensitive_value")

        if (
            not data_path
            or target_column is None
            or selected_entity_index is None
            or not sensitive_attribute
            or new_sensitive_value is None
        ):
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at '{data_path}'"}), 400

        df = pd.read_csv(data_path)

        if target_column not in df.columns or sensitive_attribute not in df.columns:
            return (
                jsonify(
                    {"error": "Target or sensitive attribute not found in dataset"}
                ),
                400,
            )

        if selected_entity_index >= len(df):
            return jsonify({"error": "Selected entity index is out of bounds"}), 400

        model = load_model()
        encoders = load_encoders()
        df_encoded = encode_categorical_columns(df.copy(), encoders)
        X = df_encoded.drop(columns=[target_column])

        selected_sample = df.iloc[selected_entity_index].copy()
        selected_sample_encoded = X.iloc[selected_entity_index].copy()
        print(selected_sample)
        original_prediction = int(model.predict([selected_sample_encoded])[0])
        counterfactual_sample = selected_sample.copy()
        counterfactual_sample[sensitive_attribute] = new_sensitive_value
        counterfactual_sample.drop(columns=[target_column])
        counterfactual_sample_encoded = encode_categorical_columns(
            pd.DataFrame([counterfactual_sample]).drop(columns=[target_column]),
            encoders,
        ).iloc[0]

        counterfactual_prediction = int(
            model.predict([counterfactual_sample_encoded])[0]
        )
        print(counterfactual_prediction)
        print(model.predict_proba([counterfactual_sample_encoded]))
        return (
            jsonify(
                {
                    "counterfactual": {
                        "flipped_attribute": sensitive_attribute,
                        "original_features": selected_sample.to_dict(),
                        "flipped_features": counterfactual_sample.to_dict(),
                        "original_prediction": original_prediction,
                        "counterfactual_prediction": counterfactual_prediction,
                    }
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from scipy.optimize import minimize
import numpy as np


def load_target_label_encoder():
    if not os.path.exists(TARGET_LABEL_ENCODER_FILE):
        raise FileNotFoundError("Target label encoder file not found.")
    with open(TARGET_LABEL_ENCODER_FILE, "rb") as f:
        return pickle.load(f)


from scipy.optimize import differential_evolution

import random

import random
import numpy as np

from scipy.optimize import dual_annealing


@counterfactual_bp.route("/recourse", methods=["POST"])
def generate_actionable_recourse():
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        selected_entity_index = request.json.get("selected_entity_index") - 1
        actionable_features = request.json.get("actionable_features")
        target_prediction = request.json.get("target_prediction")

        if not all([data_path, target_column, actionable_features, target_prediction]):
            return jsonify({"error": "Missing required parameters"}), 400
        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at '{data_path}'"}), 400

        df = pd.read_csv(data_path)

        model = load_model()
        encoders = load_encoders()
        target_encoder = load_target_label_encoder()

        df[target_column] = target_encoder.transform(df[target_column])

        encoded_target_prediction = target_encoder.transform([target_prediction])[0]

        numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        df_encoded = encode_categorical_columns(
            df.drop(columns=[target_column]), encoders
        )

        x_original = df_encoded.iloc[selected_entity_index].values

        init_prob = model.predict_proba([x_original])[0][encoded_target_prediction]

        actionable_indices = [
            df_encoded.columns.get_loc(feat) for feat in actionable_features
        ]

        max_attempts = 5000
        best_attempt = None
        best_flip_score = float("inf")

        for _ in range(max_attempts):
            new_values = x_original.copy()

            for index in actionable_indices:
                if df_encoded.columns[index] in numerical_cols:
                    new_values[index] += np.random.uniform(
                        -0.5, 0.5
                    )  # Small, efficient adjustments
                else:
                    unique_categories = df_encoded.iloc[:, index].unique()
                    new_values[index] = random.choice(unique_categories)

            prob = model.predict_proba([new_values])[0][encoded_target_prediction]

            if prob < init_prob:
                best_attempt = new_values
                break

            flip_score = 1 - prob
            if flip_score > best_flip_score:
                best_flip_score = flip_score
                best_attempt = new_values

        if best_attempt is not None:
            decoded_entity = df_encoded.iloc[selected_entity_index].copy()
            for idx in actionable_indices:
                decoded_entity.iloc[idx] = best_attempt[idx]

            decoded_entity[numerical_cols] = scaler.inverse_transform(
                [decoded_entity[numerical_cols].values]
            )[0]

            for col in categorical_cols:
                decoded_entity[col] = encoders[col].inverse_transform(
                    [int(decoded_entity[col])]
                )[0]

            suggested_changes = {
                feat: decoded_entity[feat] for feat in actionable_features
            }
            return jsonify({"recourse": suggested_changes}), 200

        return (
            jsonify(
                {
                    "error": "Failed to find a valid configuration after multiple attempts"
                }
            ),
            500,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
