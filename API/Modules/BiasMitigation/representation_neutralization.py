import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.api.layers import Dense, Dropout, BatchNormalization, Input
from keras.api.models import Model
from keras.api.losses import binary_crossentropy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Modules.DataAnalysis.GroupFairness.notions import (
    conditional_statistical_parity,
    conditional_use_accuracy_equality,
    equal_negative_predictive_value,
    equal_opportunity,
    overall_accuracy_equality,
    predictive_equality,
    statistical_parity,
    treatment_equality,
    balance_for_positive_class,
    equalized_odds,
    balance_for_negative_class,
)


class RepresentationNeutralizationModel:
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

    def neutralization_loss(
        self,
        y_true,
        y_pred,
        inputs,
        protected_column_idx,
        lambda_binary,
        lambda_neutralize,
    ):
        y_pred_squeezed = tf.squeeze(y_pred, axis=1)
        binary_loss = binary_crossentropy(y_true, y_pred_squeezed)

        embedding = self.bottleneck_extractor(inputs)

        neutralization_loss = tf.reduce_mean(
            tf.abs(
                tf.reduce_mean(embedding, axis=0)
                - tf.reduce_mean(embedding, axis=0, keepdims=True)
            )
        )

        total_loss = (
            lambda_binary * binary_loss + lambda_neutralize * neutralization_loss
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
        lambda_neutralize,
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
                    loss = self.neutralization_loss(
                        labels_batch,
                        predictions,
                        inputs_batch,
                        protected_column_idx,
                        lambda_binary,
                        lambda_neutralize,
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
            "f1_score": f1_score(y_test, predictions_binary),
            "auc": roc_auc_score(y_test, predictions),
        }


def apply_representation_neutralization(
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

    if privileged_classes is None:
        raise ValueError(
            "privileged_classes and unprivileged_classes must be provided."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    data[sensitive_column] = (data[sensitive_column] == privileged_classes).astype(int)

    split_index = int(len(data) * ((1 - test_split_percent)))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train, y_train = (
        train_data.drop(columns=[target_column]),
        train_data[target_column],
    )
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_test = pd.to_numeric(y_test, errors="coerce")

    y_train_encoded = (y_train > 0.5).astype(int)
    y_test_encoded = (y_test > 0.5).astype(int)

    feature_encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        feature_encoders[col] = le

    X_train_protected = X_train[sensitive_column]
    X_test_protected = X_test[sensitive_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train_encoded, test_size=0.001
    )
    X_train_protected = X_train[sensitive_column]
    X_val_protected = X_val[sensitive_column]

    model = RepresentationNeutralizationModel(X_train.shape[1])

    start_time = time.time()
    protected_column_idx = list(X_train.columns).index(sensitive_column)

    model.train(
        5,
        256,
        X_train,
        y_train,
        X_train_protected,
        X_val,
        y_val,
        X_val_protected,
        lambda_binary=0.9,
        lambda_neutralize=0.1,
        protected_column_idx=protected_column_idx,
    )

    end_time = time.time()
    total_time = end_time - start_time

    with open("representation_neutralization_model.pkl", "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize("representation_neutralization_model.pkl")

    test_preds = model.model.predict(X_test)
    test_preds = (test_preds > 0.5).astype(int)

    cm = confusion_matrix(y_test_encoded, test_preds)
    tn, fp, fn, tp = cm.ravel()

    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test_encoded, sensitive_column: X_test_protected}
    )
    predictions_with_sensitive = pd.DataFrame(
        {"predictions": test_preds, sensitive_column: X_test_protected}
    )

    metrics = {
        "accuracy": accuracy_score(y_test_encoded, test_preds),
        "precision": precision_score(y_test_encoded, test_preds),
        "recall": recall_score(y_test_encoded, test_preds),
        "f1_score": f1_score(y_test_encoded, test_preds),
        "auc": roc_auc_score(y_test_encoded, test_preds),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "training_time": total_time,
        "model_size": model_size,
        "conditional_statistical_parity": conditional_statistical_parity(
            y_test_with_sensitive, predictions_with_sensitive
        ),
        "conditional_use_accuracy_equality": conditional_use_accuracy_equality(
            y_test_with_sensitive, predictions_with_sensitive
        ),
        "equal_opportunity": equal_opportunity(
            y_test_with_sensitive, predictions_with_sensitive
        ),
        "equalized_odds": equalized_odds(
            y_test_with_sensitive, predictions_with_sensitive
        ),
    }

    return metrics
