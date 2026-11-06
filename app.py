"""Web dashboard for the churn predictor.

A designed UI over the real model: single-customer predictions,
model insights from the actual training run, and CSV batch scoring.
The prediction logic (predict_one) is shared with api.py - this file
adds presentation only. Run from the repo root:

    python app.py    # http://localhost:5000

Needs data/churn.csv and models/churn_model.pkl (fresh clone:
python data/generate.py && python train.py first).
"""

import io
import json
import os
import uuid

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

if not os.path.exists("models/churn_model.pkl"):
    raise SystemExit(
        "model not found - run `python data/generate.py` then `python train.py` first"
    )

from api import THRESHOLD, predict_one  # noqa: E402

import features  # noqa: E402

app = Flask(__name__)

# reports/metrics.json is pushed to git but is only generated after the
# data exists - a fresh clone degrades to a setup hint instead of crashing
try:
    with open("reports/metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = None

REQUIRED = features.NUMERIC + features.CATEGORICAL + features.BINARY

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

# figures are regenerated locally by eda.py / train.py; the PNGs are never
# pushed to git (binary files), so the page must degrade when they're absent
FIGURES = [
    ("churn_by_contract.png", "Churn rate by contract type (synthetic data)", "eda"),
    ("churn_by_tenure.png", "Churn rate by tenure bucket (synthetic data)", "eda"),
    (
        "churn_by_support_calls.png",
        "Churn rate vs number of support calls (synthetic data)",
        "eda",
    ),
    ("correlation.png", "Correlation heatmap of the numeric columns", "eda"),
    ("cm_logreg.png", None, "train"),
    ("roc_logreg.png", None, "train"),
    ("cm_random_forest.png", None, "train"),
    ("roc_random_forest.png", None, "train"),
    ("cm_gradient_boosting.png", None, "train"),
    ("roc_gradient_boosting.png", None, "train"),
]


def _figures():
    out = []
    for fname, caption, source in FIGURES:
        if caption is None:
            key = fname[3:-4] if fname.startswith("cm_") else fname[4:-4]
            m = METRICS["models"][key] if METRICS else {}
            label = MODEL_LABELS.get(key, key).lower()
            if fname.startswith("cm_"):
                caption = (
                    f"Confusion matrix - {label}, tuned threshold "
                    f"{m.get('threshold', '?')}"
                )
            else:
                auc = m.get("roc_auc")
                caption = f"ROC curve - {label}" + (
                    f", AUC {auc:.2f}" if auc is not None else ""
                )
        out.append(
            {
                "file": fname,
                "caption": caption,
                "exists": os.path.exists(os.path.join("static", "figures", fname)),
                "source": source,
            }
        )
    return out

# downloads for the batch page: token -> scored dataframe
_scored = {}


def _form_record(form) -> dict:
    def num(key, cast):
        return cast(form.get(key, "0") or 0)

    return {
        "tenure": num("tenure", int),
        "monthly_charges": num("monthly_charges", float),
        "total_charges": num("total_charges", float),
        "support_calls": num("support_calls", int),
        "contract_type": form.get("contract_type", "Month-to-month"),
        "internet_service": form.get("internet_service", "Fiber optic"),
        "payment_method": form.get("payment_method", "Electronic check"),
        "senior_citizen": 1 if form.get("senior_citizen") else 0,
        "partner": 1 if form.get("partner") else 0,
        "dependents": 1 if form.get("dependents") else 0,
    }


def _verdict(probability: float) -> str:
    return "Likely to churn" if probability >= THRESHOLD else "Likely to stay"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    record = None
    error = None
    if request.method == "POST":
        try:
            record = _form_record(request.form)
            out = predict_one(record)
        except (ValueError, KeyError):
            # _form_record raises ValueError on a non-numeric field; the
            # browser's type=number inputs normally prevent it, but a
            # hand-built POST (curl, tweaked HTML) used to 500 here
            error = "Tenure, charges and support calls must be numbers."
            out = None
        if out is not None:
            result = {
                "record": record,
                "probability": out["churn_probability"],
                "verdict": _verdict(out["churn_probability"]),
                "threshold": THRESHOLD,
            }
    if METRICS:
        model_name = MODEL_LABELS.get(METRICS["best_model"], METRICS["best_model"])
    else:
        model_name = "the trained model"
    return render_template(
        "index.html",
        result=result,
        record=record,
        error=error,
        model_name=model_name,
        threshold=THRESHOLD,
    )


@app.route("/predict", methods=["POST"])
def predict():
    # same JSON contract as api.py
    try:
        return jsonify(predict_one(request.get_json(force=True)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "invalid input, check field names and types"}), 400


@app.route("/explore")
def explore():
    if METRICS is None:
        # fresh clone: data not generated yet, table degrades to a hint
        return render_template("explore.html", models=None, figures=_figures())

    models = []
    for key, m in METRICS["models"].items():
        models.append(
            {
                "label": MODEL_LABELS.get(key, key),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "threshold": m["threshold"],
            }
        )
    models.sort(key=lambda r: r["f1"], reverse=True)
    best_model = MODEL_LABELS.get(METRICS["best_model"], METRICS["best_model"])

    return render_template(
        "explore.html",
        models=models,
        figures=_figures(),
        best_model=best_model,
        n_test=METRICS["n_test"],
        test_churn_rate=METRICS["test_churn_rate"],
    )


@app.route("/batch", methods=["GET", "POST"])
def batch():
    result = None
    error = None
    if request.method == "POST":
        upload = request.files.get("csv")
        if upload is None or not upload.filename:
            error = "Pick a CSV file first."
        else:
            try:
                df = pd.read_csv(upload)
            except Exception:
                error = "Could not read that file as CSV."
            else:
                missing = [c for c in REQUIRED if c not in df.columns]
                if missing:
                    error = f"Missing columns: {', '.join(missing)}"
                elif df.empty:
                    error = "The CSV has no rows."
                else:
                    probs = []
                    for _, row in df.iterrows():
                        out = predict_one({c: row[c] for c in REQUIRED})
                        probs.append(out["churn_probability"])
                    scored = df.copy()
                    scored["churn_probability"] = [round(p, 4) for p in probs]
                    scored["prediction"] = [
                        "churn" if p >= THRESHOLD else "stay" for p in probs
                    ]
                    token = uuid.uuid4().hex
                    _scored[token] = scored
                    flagged = int((scored["prediction"] == "churn").sum())
                    result = {
                        "n": len(scored),
                        "flagged": flagged,
                        "token": token,
                        "rows": scored.head(25).to_dict("records"),
                        "columns": list(scored.columns),
                    }
    return render_template("batch.html", result=result, error=error, threshold=THRESHOLD)


@app.route("/batch/sample")
def batch_sample():
    # 30 real rows from the dataset, kept with their labels so you can
    # eyeball the model's calls against the true outcome
    df = pd.read_csv("data/churn.csv", nrows=30)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_sample_30.csv",
    )


@app.route("/batch/download/<token>")
def batch_download(token):
    scored = _scored.get(token)
    if scored is None:
        return "That download link has expired - score the file again.", 404
    buf = io.StringIO()
    scored.to_csv(buf, index=False)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="churn_scored.csv",
    )


if __name__ == "__main__":
    app.run(debug=True)
