"""Train and compare churn models.

Trains three classifiers on the preprocessed data, prints
accuracy / precision / recall / F1 / ROC-AUC for each, saves a
confusion matrix + ROC curve per model to reports/figures/, and
pickles the best model (by F1) to models/churn_model.pkl along with
the fitted preprocessor.

Run: python train.py [--data data/churn.csv]
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

import features

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def tune_threshold(model, X_train, y_train) -> float:
    """Pick the decision threshold that maximizes F1.

    Done with 3-fold CV predictions on the TRAIN set only, so the test
    set stays untouched for the final evaluation. The default 0.5
    threshold under-predicts churn here (only ~28% of customers churn),
    so this is worth doing.
    """
    proba = cross_val_predict(model, X_train, y_train, cv=3, method="predict_proba")[
        :, 1
    ]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.15, 0.85, 0.05):
        f1 = f1_score(y_train, proba >= t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return round(float(best_t), 2)


def evaluate(name, model, X_train, y_train, X_test, y_test):
    threshold = tune_threshold(model, X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = proba >= threshold
    scores = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "threshold": threshold,
    }
    print(f"\n{name} (threshold={threshold})")
    for k, v in scores.items():
        if k != "threshold":
            print(f"  {k:10s} {v:.4f}")

    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/cm_{name}.png")
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title(f"ROC curve - {name}")
    plt.tight_layout()
    plt.savefig(f"reports/figures/roc_{name}.png")
    plt.close()

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    pre, X_train, X_test, y_train, y_test = features.train_test(args.data)
    print(f"train: {X_train.shape[0]} rows, test: {X_test.shape[0]} rows")

    results = {}
    trained = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(name, model, X_train, y_train, X_test, y_test)
        trained[name] = model

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nbest model by F1: {best} ({results[best]['f1']:.4f})")

    # save the winner + the fitted preprocessor + tuned threshold together
    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(
            {
                "preprocessor": pre,
                "model": trained[best],
                "name": best,
                "threshold": results[best]["threshold"],
            },
            f,
        )
    print("saved -> models/churn_model.pkl")


if __name__ == "__main__":
    main()
