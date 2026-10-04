"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

import features


@pytest.fixture
def sample_df():
    # tiny hand-made frame covering every column type
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "tenure": [5, 40, 12, 60],
            "monthly_charges": [80.0, 45.5, 100.0, 30.0],
            "total_charges": [400.0, 1820.0, 1200.0, 1800.0],
            "contract_type": ["Month-to-month", "Two year", "One year", "Month-to-month"],
            "internet_service": ["Fiber optic", "DSL", "Fiber optic", "No internet"],
            "payment_method": [
                "Electronic check",
                "Credit card",
                "Bank transfer",
                "Mailed check",
            ],
            "senior_citizen": [0, 1, 0, 0],
            "partner": [1, 0, 1, 1],
            "dependents": [0, 0, 1, 0],
            "support_calls": [3, 0, 1, 2],
            "churn": [1, 0, 1, 0],
        }
    )


def test_split_features_target(sample_df):
    X, y = features.split_features_target(sample_df)
    assert "churn" not in X.columns
    assert "customer_id" not in X.columns
    assert list(y) == [1, 0, 1, 0]


def test_preprocessor_no_nans(sample_df):
    X, _ = features.split_features_target(sample_df)
    pre = features.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert not np.isnan(Xt).any()


def test_preprocessor_output_width(sample_df):
    # 4 numeric + one-hot(contract 3 + internet 3 + payment 4) + 3 binary
    X, _ = features.split_features_target(sample_df)
    pre = features.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[1] == 4 + 10 + 3


def test_full_pipeline_predicts(sample_df):
    from sklearn.linear_model import LogisticRegression

    X, y = features.split_features_target(sample_df)
    pipe = features.full_pipeline(LogisticRegression())
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert set(preds) <= {0, 1}
    assert len(preds) == len(y)


def test_feature_names_match_width(sample_df):
    X, _ = features.split_features_target(sample_df)
    pre = features.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert len(features.feature_names(pre)) == Xt.shape[1]
