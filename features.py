"""Preprocessing for the churn dataset.

Everything goes through a sklearn ColumnTransformer so the same
transformations apply at train time and at prediction time (the API
reuses this). Numeric columns get standardized, categoricals get
one-hot encoded, target column is split off, and the split is
stratified so train/test keep the same churn rate.

No leakage: the split happens on raw features, the transformer is
fit on train only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
ID_COL = "customer_id"

NUMERIC = ["tenure", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL = ["contract_type", "internet_service", "payment_method"]
BINARY = ["senior_citizen", "partner", "dependents"]  # already 0/1, pass through


def load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )


def train_test(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """Load, split (stratified), and fit the preprocessor on train only."""
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_raw(
        csv_path, test_size=test_size, seed=seed
    )
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train_raw)
    X_test_t = pre.transform(X_test_raw)
    return pre, X_train_t, X_test_t, y_train, y_test


def train_test_split_raw(csv_path: str, test_size: float = 0.2, seed: int = 42):
    """Stratified split on the RAW features (pre-transform).

    evaluate.py needs the raw test features so it can run them through
    the saved preprocessor itself.
    """
    df = load(csv_path)
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def feature_names(pre: ColumnTransformer) -> list:
    """Names of the transformed features (for inspection / debugging)."""
    num = NUMERIC
    cat = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    return num + cat + BINARY


def full_pipeline(model) -> Pipeline:
    """Preprocessing + model in one pipeline (handy for the API)."""
    return Pipeline([("pre", build_preprocessor()), ("model", model)])
