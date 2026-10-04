"""Prediction API for the churn model.

POST /predict  - JSON customer record -> {"churn_probability": 0.73, "prediction": "churn"}
GET  /         - a small HTML form so you can try it in the browser

Loads models/churn_model.pkl (fitted preprocessor + best model) once
at startup. Run: python api.py  (then open http://localhost:5000)

Example:
  curl -X POST localhost:5000/predict -H "Content-Type: application/json" -d '{
    "tenure": 3, "monthly_charges": 95.5, "total_charges": 286.5,
    "contract_type": "Month-to-month", "internet_service": "Fiber optic",
    "payment_method": "Electronic check", "senior_citizen": 0,
    "partner": 0, "dependents": 0, "support_calls": 4}'
"""

import pickle

import pandas as pd
from flask import Flask, jsonify, request

import features

app = Flask(__name__)

with open("models/churn_model.pkl", "rb") as f:
    _bundle = pickle.load(f)
PRE = _bundle["preprocessor"]
MODEL = _bundle["model"]
THRESHOLD = _bundle.get("threshold", 0.5)

# the fields /predict expects - must match features.py
REQUIRED = (
    features.NUMERIC + features.CATEGORICAL + features.BINARY
)


def predict_one(record: dict) -> dict:
    missing = [c for c in REQUIRED if c not in record]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    df = pd.DataFrame([{c: record[c] for c in REQUIRED}])
    proba = float(MODEL.predict_proba(PRE.transform(df))[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "prediction": "churn" if proba >= THRESHOLD else "stay",
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        return jsonify(predict_one(request.get_json(force=True)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:  # bad types etc. - don't leak tracebacks
        return jsonify({"error": "invalid input, check field names and types"}), 400


FORM_HTML = """<!doctype html>
<html><head><title>Churn predictor</title></head>
<body style="font-family: sans-serif; max-width: 500px; margin: 40px auto;">
<h2>Will this customer churn?</h2>
<form id="f">
  Tenure (months): <input name="tenure" type="number" value="6"><br><br>
  Monthly charges: <input name="monthly_charges" type="number" step="0.01" value="89.99"><br><br>
  Total charges: <input name="total_charges" type="number" step="0.01" value="540"><br><br>
  Support calls: <input name="support_calls" type="number" value="3"><br><br>
  Contract: <select name="contract_type">
    <option>Month-to-month</option><option>One year</option><option>Two year</option>
  </select><br><br>
  Internet: <select name="internet_service">
    <option>Fiber optic</option><option>DSL</option><option>No internet</option>
  </select><br><br>
  Payment: <select name="payment_method">
    <option>Electronic check</option><option>Mailed check</option>
    <option>Bank transfer</option><option>Credit card</option>
  </select><br><br>
  <label><input name="senior_citizen" type="checkbox"> Senior citizen</label><br>
  <label><input name="partner" type="checkbox"> Has partner</label><br>
  <label><input name="dependents" type="checkbox"> Has dependents</label><br><br>
  <button type="submit">Predict</button>
</form>
<h3 id="out"></h3>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const num = (k) => parseFloat(fd.get(k));
  const body = {
    tenure: num('tenure'), monthly_charges: num('monthly_charges'),
    total_charges: num('total_charges'), support_calls: num('support_calls'),
    contract_type: fd.get('contract_type'), internet_service: fd.get('internet_service'),
    payment_method: fd.get('payment_method'),
    senior_citizen: fd.get('senior_citizen') ? 1 : 0,
    partner: fd.get('partner') ? 1 : 0,
    dependents: fd.get('dependents') ? 1 : 0,
  };
  const r = await fetch('/predict', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  document.getElementById('out').textContent =
    `Prediction: ${j.prediction} (probability ${j.churn_probability})`;
};
</script>
</body></html>
"""


@app.route("/")
def index():
    return FORM_HTML


if __name__ == "__main__":
    app.run(debug=True)
