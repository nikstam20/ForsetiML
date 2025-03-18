from flask import Blueprint, request, jsonify
import os
from Modules.BiasMitigation.adversarial_debiasing import adversarial_debiasing
from Modules.BiasMitigation.disparate_impact_remover import disparate_impact_remover

bias_mitigation_bp = Blueprint("bias_mitigation", __name__)


@bias_mitigation_bp.route("/apply_disparate_impact_remover", methods=["POST"])
def apply_disparate_impact_remover_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        repair_level = request.json.get("repair_level", 1.0)
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)
        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = disparate_impact_remover(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            repair_level=repair_level,
            target_column=target_attribute,
            test_split_percent=test_size,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Disparate Impact Remover applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.adversarial_debiasing import adversarial_debiasing


@bias_mitigation_bp.route("/apply_adversarial_debiasing", methods=["POST"])
def apply_adversarial_debiasing_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)
        num_epochs = request.json.get("num_epochs", 20)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = adversarial_debiasing(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
            num_epochs=num_epochs,
        )

        return (
            jsonify(
                {
                    "message": "Adversarial Debiasing applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.egr import apply_egr


@bias_mitigation_bp.route("/apply_exponentiated_gradient_reduction", methods=["POST"])
def apply_exponentiated_gradient_reduction_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_egr(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Exponentiated Gradient Reduction applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.gerry_fair import apply_gerry_fair


@bias_mitigation_bp.route("/apply_gerry_fair_classifier", methods=["POST"])
def apply_gerry_fair_classifier_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        c = request.json.get("c", 0.01)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_gerry_fair(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=42,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "GerryFair Classifier applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.art import apply_art


@bias_mitigation_bp.route("/apply_adversarially_robust_training", methods=["POST"])
def apply_adversarially_robust_training_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_art(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Adversarially Robust Training applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.fairvic import apply_fairvic


@bias_mitigation_bp.route("/apply_fair_vic", methods=["POST"])
def apply_fair_vic_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_fairvic(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify({"message": "FairVIC applied successfully", "metrics": metrics}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.representation_neutralization import (
    apply_representation_neutralization,
)


@bias_mitigation_bp.route("/apply_representation_neutralization", methods=["POST"])
def apply_representation_neutralization_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_representation_neutralization(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Representation Neutralization applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.deterministic_reranking import deterministic_reranking


@bias_mitigation_bp.route("/apply_deterministic_reranking", methods=["POST"])
def apply_deterministic_reranking_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = deterministic_reranking(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Deterministic Reranking applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.meta_fair import apply_meta_fair_classifier


@bias_mitigation_bp.route("/apply_meta_fair_classification", methods=["POST"])
def apply_meta_fair_classification_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        privileged_classes = request.json.get("privileged_group")
        unprivileged_classes = request.json.get("unprivileged_group")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_meta_fair_classifier(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            privileged_classes=privileged_classes,
            unprivileged_classes=unprivileged_classes,
            random_state=random_state,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Meta Fair Classification applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.relabelling import relabel_data


@bias_mitigation_bp.route("/apply_relabel_data", methods=["POST"])
def apply_relabel_data_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = relabel_data(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {"message": "Relabel Data applied successfully", "metrics": metrics}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.eq_odd_postproc import equalized_odds_postprocessing


@bias_mitigation_bp.route("/apply_equalized_odds_postprocessing", methods=["POST"])
def apply_eq_odds_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = equalized_odds_postprocessing(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Equalized odds postprocessing applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.cal_eq_odd_postproc import (
    calibrated_equalized_odds_postprocessing,
)


@bias_mitigation_bp.route(
    "/apply_calibrated_equalized_odds_postprocessing", methods=["POST"]
)
def apply_cal_eq_odds_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = calibrated_equalized_odds_postprocessing(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Calibrated equalized odds postprocessing applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.reject_option_classification import (
    reject_option_classification,
)


@bias_mitigation_bp.route("/apply_reject_option_classification", methods=["POST"])
def apply_reject_option_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = reject_option_classification(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Reject Option Classification applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.feature_importance_suppression import (
    feature_importance_suppression,
)


@bias_mitigation_bp.route("/apply_feature_importance_suppression", methods=["POST"])
def apply_feature_importance_suppression_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        importance_threshold = request.json.get("importance_threshold", 0.1)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = feature_importance_suppression(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            importance_threshold=importance_threshold,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Feature Importance Suppression applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.optimized_preprocessing import optimized_preprocessing


@bias_mitigation_bp.route("/apply_optimized_preprocessing", methods=["POST"])
def apply_optimized_preprocessing_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)
        test_size = request.json.get("test_size", 0.2)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = optimized_preprocessing(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Optimized Preprocessing applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.label_flipping import flip_labels


@bias_mitigation_bp.route("/apply_label_flipping", methods=["POST"])
def apply_label_flipping_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        flip_rate = request.json.get("flip_rate", 0.1)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = flip_labels(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            flip_rate=flip_rate,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {"message": "Label Flipping applied successfully", "metrics": metrics}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.predujice_remover import apply_prejudice_remover


@bias_mitigation_bp.route("/apply_prejudice_remover", methods=["POST"])
def apply_prejudice_remover_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        eta = request.json.get("eta", 1.0)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)
        test_size = request.json.get("test_size", 0.2)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = apply_prejudice_remover(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            eta=eta,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Prejudice Remover applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigation.correlation_suppression import correlation_suppression


@bias_mitigation_bp.route("/apply_correlation_suppression", methods=["POST"])
def apply_correlation_suppression_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        correlation_threshold = request.json.get("correlation_threshold", 0.5)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = correlation_suppression(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            correlation_threshold=correlation_threshold,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Correlation Suppression applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from API.Modules.BiasMitigation.prevalence_sampling import prevalence_sampling


@bias_mitigation_bp.route("/apply_prevalence_sampling", methods=["POST"])
def apply_prevalence_sampling_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = correlation_suppression(
            data_path=data_path,
            target_column=target_attribute,
            sensitive_column=sensitive_attribute,
            random_state=random_state,
            privileged_classes=privileged_group,
            unprivileged_classes=unprivileged_group,
        )

        return (
            jsonify(
                {
                    "message": "Prevalence Sampling applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigationRegression.disparate_impact_remover import dir_regression


@bias_mitigation_bp.route("/apply_disparate_impact_remover_reg", methods=["POST"])
def apply_disparate_impact_remover_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        repair_level = request.json.get("repair_level", 1.0)
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)
        privileged_group = request.json.get("privileged_group", None)
        unprivileged_group = request.json.get("unprivileged_group", None)
        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = dir_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            repair_level=repair_level,
            target_column=target_attribute,
            test_split_percent=test_size,
        )

        return (
            jsonify(
                {
                    "message": "Disparate Impact Remover applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigationRegression.sfr import feature_removal_regression


@bias_mitigation_bp.route("/apply_feature_removal_reg", methods=["POST"])
def apply_feature_removal_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = feature_removal_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            target_column=target_attribute,
            test_split_percent=test_size,
            random_state=random_state,
        )

        return (
            jsonify(
                {"message": "Feature Removal applied successfully", "metrics": metrics}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigationRegression.reweighting import reweighting_regression


@bias_mitigation_bp.route("/apply_reweighting_reg", methods=["POST"])
def apply_reweighting_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = reweighting_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            target_column=target_attribute,
            test_split_percent=test_size,
            random_state=random_state,
        )

        return (
            jsonify(
                {"message": "Reweighting applied successfully", "metrics": metrics}
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigationRegression.adversarial_debiasing import (
    adversarial_debiasing_regression,
)
from Modules.BiasMitigationRegression.prejudice_remover import (
    prejudice_remover_regression,
)


@bias_mitigation_bp.route("/apply_adversarial_debiasing_reg", methods=["POST"])
def apply_adversarial_debiasing_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        adversary_weight = request.json.get("adversary_weight", 0.1)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = adversarial_debiasing_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            target_column=target_attribute,
            test_split_percent=test_size,
            adversary_weight=adversary_weight,
            random_state=random_state,
        )

        return (
            jsonify(
                {
                    "message": "Adversarial Debiasing applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bias_mitigation_bp.route("/apply_prejudice_remover_reg", methods=["POST"])
def apply_prejudice_remover_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        fairness_penalty = request.json.get("fairness_penalty", 0.1)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = prejudice_remover_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            target_column=target_attribute,
            test_split_percent=test_size,
            fairness_penalty=fairness_penalty,
            random_state=random_state,
        )

        return (
            jsonify(
                {
                    "message": "Prejudice Remover applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


from Modules.BiasMitigationRegression.calibrated_equalized_odds import (
    calibrated_equalized_odds_regression,
)


@bias_mitigation_bp.route(
    "/apply_calibrated_equalized_odds_postprocessing_reg", methods=["POST"]
)
def apply_calibrated_equalized_odds_reg_endpoint():
    try:
        data_path = request.json.get("data_path")
        sensitive_attribute = request.json.get("sensitive_attribute")
        target_attribute = request.json.get("target_attribute")
        test_size = request.json.get("test_size", 0.2)
        calibration_strength = request.json.get("calibration_strength", 0.1)
        random_state = request.json.get("random_state", 42)

        if not data_path or not sensitive_attribute or not target_attribute:
            return jsonify({"error": "Missing required parameters"}), 400

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        metrics = calibrated_equalized_odds_regression(
            data_path=data_path,
            sensitive_column=sensitive_attribute,
            target_column=target_attribute,
            test_split_percent=test_size,
            calibration_strength=calibration_strength,
            random_state=random_state,
        )

        return (
            jsonify(
                {
                    "message": "Calibrated Equalized Odds applied successfully",
                    "metrics": metrics,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
