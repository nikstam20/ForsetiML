from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd


def preprocess_features(X):
    """
    Preprocesses the features matrix to encode categorical variables and scale numeric features.

    Args:
        X (pd.DataFrame): The input features dataset containing both categorical and numeric columns.

    Returns:
        pd.DataFrame: A transformed DataFrame with encoded categorical columns and scaled numeric columns.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    categorical_transformer = OneHotEncoder(drop="if_binary", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return pd.DataFrame(
        preprocessor.fit_transform(X), columns=preprocessor.get_feature_names_out()
    )


def consistency(y_true, X, sample_size=5000, random_state=42):
    """
    Computes the consistency metric by measuring prediction differences for similar data points.

    Args:
        y_true (pd.Series or np.ndarray): The true target values corresponding to the features.
        X (pd.DataFrame): The input features dataset containing both categorical and numeric columns.
        sample_size (int, optional): The number of samples to use for pairwise distance calculation. Defaults to 5000.
        random_state (int, optional): Random seed for reproducibility when sampling. Defaults to 42.

    Returns:
        float: The consistency metric value, where higher values indicate more consistent predictions for similar data points.
    """
    if len(X) > sample_size:
        X, y_true = X.sample(n=sample_size, random_state=random_state), y_true.sample(
            n=sample_size, random_state=random_state
        )

    X = preprocess_features(X)
    distances = pairwise_distances(X)
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    mean_consistency = np.mean(
        [
            np.std(y_true[distances[i] < np.percentile(distances[i], 5)])
            for i in range(len(X))
        ]
    )

    return 1 - mean_consistency
