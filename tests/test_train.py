"""Tests for train.py's metrics.json output (tests/test_train.py)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train


def test_write_metrics_matches_dashboard_shape(tmp_path, monkeypatch):
    # train.py runs from the repo root; point the write at a temp dir
    monkeypatch.chdir(tmp_path)
    results = {
        "logreg": {
            "accuracy": 0.77,
            "precision": 0.58,
            "recall": 0.64,
            "f1": 0.61,
            "roc_auc": 0.81,
            "threshold": 0.35,
        },
        "gradient_boosting": {
            "accuracy": 0.73,
            "precision": 0.51,
            "recall": 0.76,
            "f1": 0.61,
            "roc_auc": 0.80,
            "threshold": 0.25,
        },
    }
    train.write_metrics(
        results, "logreg", 1000, 0.278, "data/churn.csv (synthetic, 5000 rows)"
    )
    with open(tmp_path / "reports" / "metrics.json") as f:
        payload = json.load(f)
    # every key the dashboard reads must be there, as plain JSON numbers
    assert payload["best_model"] == "logreg"
    assert payload["n_test"] == 1000
    assert payload["test_churn_rate"] == 0.278
    assert payload["dataset"] == "data/churn.csv (synthetic, 5000 rows)"
    assert payload["models"]["logreg"]["threshold"] == 0.35
    assert payload["models"]["gradient_boosting"]["f1"] == 0.61
    # the explore page sorts on f1 and formats roc_auc - both must be floats
    assert isinstance(payload["models"]["logreg"]["roc_auc"], float)
