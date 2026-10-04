"""Evaluate the saved model on the held-out test set.

Loads models/churn_model.pkl (model + fitted preprocessor), rebuilds
the same stratified split train.py used, and prints the full
classification report. Mostly a sanity check that the saved artifact
actually works - and a quick way to re-check metrics without retraining.

Run: python evaluate.py [--data data/churn.csv]
"""

import argparse
import pickle

from sklearn.metrics import classification_report

import features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    args = parser.parse_args()

    with open("models/churn_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    print(f"loaded model: {bundle['name']}")

    # same split as training (same seed -> same test set)
    _, X_test_raw, _, y_test = features.train_test_split_raw(args.data)

    X_test = bundle["preprocessor"].transform(X_test_raw)
    proba = bundle["model"].predict_proba(X_test)[:, 1]
    threshold = bundle.get("threshold", 0.5)
    pred = proba >= threshold
    print(f"(decision threshold: {threshold})")

    print(classification_report(y_test, pred, target_names=["stayed", "churned"]))


if __name__ == "__main__":
    main()
