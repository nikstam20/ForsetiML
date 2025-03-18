import os
import pickle
import pandas as pd
import tempfile
import sys
import subprocess

python_executable = sys.executable
import time
from aif360.algorithms import Transformer

import numpy as np
import tensorflow as tf
from keras.api.layers import Dense, Activation, Dropout, Input
from tensorflow.python.keras import Sequential
from keras.src.layers.normalization.batch_normalization import BatchNormalization

from keras.api.regularizers import L2, L1L2
from keras.api.models import Model
from keras.api.losses import binary_crossentropy
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


class FairModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.create_model()
        self.bottleneck_extractor = Model(
            inputs=self.model.input, outputs=self.model.get_layer("bottleneck").output
        )
        self.protected_column_idx = None

    def create_model(self):
        input_layer = Input(shape=(self.input_shape,))
        x = Dense(
            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(
            64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        bottleneck = Dense(2, name="bottleneck")(x)

        x = Dense(
            64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(bottleneck)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(
            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)
        )(x)
        x = BatchNormalization()(x)

        output_layer = Dense(1, activation="sigmoid")(x)

        return Model(inputs=input_layer, outputs=output_layer)

    def fairvic_loss(
        self,
        y_true,
        y_pred,
        inputs,
        protected_column_idx,
        lambda_binary,
        lambda_cov,
        lambda_var,
        lambda_inv,
    ):
        y_pred_squeezed = tf.squeeze(y_pred, axis=1)
        binary_loss = binary_crossentropy(y_true, y_pred_squeezed)

        embedding = self.bottleneck_extractor(inputs)
        variance_loss = tf.reduce_mean(
            tf.nn.relu(1.0 - tf.math.reduce_std(embedding, axis=0))
        )

        flipped_inputs = tf.tensor_scatter_nd_update(
            inputs,
            [[i, protected_column_idx] for i in range(tf.shape(inputs)[0])],
            1 - tf.gather(inputs, protected_column_idx, axis=1),
        )
        y_flip_pred = self.model(flipped_inputs, training=True)
        invariance_loss = tf.reduce_mean(
            tf.square(y_pred_squeezed - tf.squeeze(y_flip_pred))
        )

        y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=0)
        protected_reshaped = tf.cast(inputs[:, protected_column_idx], tf.float32)
        cov_matrix = tf.reduce_mean(
            y_pred_centered * (protected_reshaped - tf.reduce_mean(protected_reshaped))
        )
        covariance_loss = tf.sqrt(tf.reduce_sum(tf.square(cov_matrix)))

        total_loss = (
            lambda_binary * binary_loss
            + lambda_inv * invariance_loss
            + lambda_cov * covariance_loss
            + lambda_var * variance_loss
        )

        return total_loss

    def train(
        self,
        epochs,
        batch_size,
        X_train,
        y_train,
        X_train_protected,
        X_val,
        y_val,
        X_val_protected,
        lambda_binary,
        lambda_cov,
        lambda_var,
        lambda_inv,
        protected_column_idx,
    ):

        self.protected_column_idx = protected_column_idx
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        for epoch in range(epochs):
            dataset = tf.data.Dataset.from_tensor_slices(
                (X_train, y_train, X_train_protected)
            ).batch(batch_size)

            for inputs_batch, labels_batch, protected_batch in dataset:
                with tf.GradientTape() as tape:
                    predictions = self.model(inputs_batch, training=True)
                    loss = self.fairvic_loss(
                        labels_batch,
                        predictions,
                        inputs_batch,
                        protected_column_idx,
                        lambda_binary,
                        lambda_cov,
                        lambda_var,
                        lambda_inv,
                    )
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        predictions_binary = (predictions > 0.5).astype(int)

        return {
            "accuracy": accuracy_score(y_test, predictions_binary),
            "precision": precision_score(y_test, predictions_binary),
            "recall": recall_score(y_test, predictions_binary),
            "f1": f1_score(y_test, predictions_binary),
            "auc": roc_auc_score(y_test, predictions),
        }


from aif360.datasets import StandardDataset
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from Modules.DataAnalysis.GroupFairness.notions import (
    conditional_statistical_parity,
    conditional_use_accuracy_equality,
    equal_negative_predictive_value,
    equal_opportunity,
    overall_accuracy_equality,
    predictive_equality,
    predictive_parity,
    statistical_parity,
    treatment_equality,
    total_fairness,
    balance_for_positive_class,
    equalized_odds,
    balance_for_negative_class,
    test_fairness,
    well_calibration,
)


def apply_fairvic(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    eta: float = 0.1,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
):
    """
    Apply Prejudice Remover and train/evaluate a model on the original and repaired datasets.

    Parameters:
    - data_path: str
        Path to the input CSV dataset.
    - target_column: str
        Name of the target column in the dataset.
    - sensitive_column: str
        Name of the sensitive attribute to debias.
    - eta: float
        Fairness regularization parameter (higher values = more fairness).
    - test_split_percent: float
        Proportion of data to be used as test set (percentage).
    - model_params: dict
        Parameters for the XGBoost model.
    - privileged_classes: list
        List of privileged classes for the sensitive attribute.
    - unprivileged_classes: list
        List of unprivileged classes for the sensitive attribute.
    - random_state: int
        Random seed for reproducibility.

    Returns:
    - metrics: dict
        Dictionary containing model performance metrics before and after repair.
    """
    if privileged_classes is None:
        raise ValueError(
            "privileged_classes and unprivileged_classes must be provided."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    data.loc[data[sensitive_column] == privileged_classes, sensitive_column] = 1
    data.loc[data[sensitive_column] == unprivileged_classes, sensitive_column] = 0
    if target_column not in data.columns or sensitive_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' or sensitive column '{sensitive_column}' not found in data."
        )
    split_index = int(len(data) * ((1 - test_split_percent)))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train, y_train = (
        train_data.drop(columns=[target_column]),
        train_data[target_column],
    )
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.fit_transform(y_test)
    shp = X_train.shape[1]
    with open("target_label_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    feature_encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        feature_encoders[col] = le
    privileged_classes = [[1]]

    def z_score_normalize(df, feature_names):
        for feature in feature_names:
            mean = df[feature].mean()
            std = df[feature].std()
            df[feature] = (df[feature] - mean) / std
        return df

    X_train = z_score_normalize(X_train, X_train.columns)
    X_test = z_score_normalize(X_test, X_test.columns)
    train_ds = StandardDataset(
        df=pd.concat([X_train, pd.Series(y_train_encoded, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1.0],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        categorical_features=X_train.select_dtypes(include=["object"]).columns,
        features_to_keep=X_train.columns.tolist(),
    )

    test_ds = StandardDataset(
        df=pd.concat([X_test, pd.Series(y_test_encoded, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1.0],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        categorical_features=X_test.select_dtypes(include=["object"]).columns,
        features_to_keep=X_test.columns.tolist(),
    )
    acc_weight = 0.1
    lambda_binary = acc_weight
    lambda_cov = lambda_var = lambda_inv = (1 - acc_weight) / 3
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_encoded, test_size=0.001
    )
    X_train_protected = X_train.loc[X_train.index, sensitive_column]
    X_val_protected = X_val.loc[X_val.index, sensitive_column]

    model = FairModel(shp)

    start_time = time.time()
    protected_column_idx = data.columns.get_loc(sensitive_column)
    model.train(
        5,
        256,
        X_train,
        y_train,
        X_train_protected,
        X_val,
        y_val,
        X_val_protected,
        lambda_binary=lambda_binary,
        lambda_cov=lambda_cov,
        lambda_var=lambda_var,
        lambda_inv=lambda_inv,
        protected_column_idx=protected_column_idx,
    )
    end_time = time.time()
    total_time = end_time - start_time

    with open("fairvic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize("pjr_model.pkl")

    test_preds = model.model.predict(X_test)
    test_preds = (test_preds > 0.5).astype(int)

    test_preds = test_preds.flatten()
    accuracy_repaired = accuracy_score(y_test_encoded, test_preds)
    cm_repaired = confusion_matrix(y_test_encoded, test_preds)
    tn, fp, fn, tp = cm_repaired.ravel()

    sensitive_column_column = test_data[sensitive_column]

    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test_encoded, f"{sensitive_column}": sensitive_column_column}
    )

    predictions_with_sensitive = pd.DataFrame(
        {"predictions": test_preds, f"{sensitive_column}": sensitive_column_column}
    )

    conditional_stat_parity = conditional_statistical_parity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    conditional_accuracy_eq = conditional_use_accuracy_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equal_neg_predictive_val = equal_negative_predictive_value(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equal_opportunity_metric = equal_opportunity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    overall_accuracy_eq = overall_accuracy_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    predictive_eq = predictive_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    statistical_parity_metric = statistical_parity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        predictions_with_sensitive[sensitive_column],
    )

    treatment_eq = treatment_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    balance_positive_class = balance_for_positive_class(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equalized_odds_metric = equalized_odds(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    balance_negative_class = balance_for_negative_class(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    precision_repaired = precision_score(y_test_encoded, test_preds)
    recall_repaired = recall_score(y_test_encoded, test_preds)
    f1_score_repaired = f1_score(y_test_encoded, test_preds)

    metrics = {
        "predictions": test_preds.tolist(),
        "accuracy": float(accuracy_repaired),
        "precision": float(precision_repaired),
        "recall": float(recall_repaired),
        "f1_score": float(f1_score_repaired),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "training_time": total_time,
        "model_size": model_size,
        "conditional_statistical_parity": conditional_stat_parity,
        "conditional_use_accuracy_equality": conditional_accuracy_eq,
        "equal_negative_predictive_value": equal_neg_predictive_val,
        "equal_opportunity": equal_opportunity_metric,
        "overall_accuracy_equality": overall_accuracy_eq,
        "predictive_equality": predictive_eq,
        "statistical_parity": statistical_parity_metric,
        "treatment_equality": treatment_eq,
        "balance_for_positive_class": balance_positive_class,
        "equalized_odds": equalized_odds_metric,
        "balance_for_negative_class": balance_negative_class,
    }

    return metrics
