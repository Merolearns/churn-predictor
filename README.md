# churn-predictor

End-to-end customer churn prediction with scikit-learn. I built this to learn the full ML workflow on one project: generating data, exploring it, engineering features, training and comparing models, evaluating properly, and serving predictions behind a small API.

**Important:** the dataset is synthetic — I generated it myself (`data/generate.py`) because real telecom churn data isn't publicly available. The churn signal is baked in from plausible drivers (short tenure, month-to-month contracts, high charges, lots of support calls), so treat the metrics as a workflow exercise, not a business result.

## How to run

```bash
pip install -r requirements.txt

python data/generate.py        # build the dataset -> data/churn.csv
python eda.py                  # explore + save plots to reports/figures/
python train.py                # train 3 models, save best to models/churn_model.pkl
python evaluate.py             # classification report on the test set
python api.py                  # prediction API on http://localhost:5000
pytest                         # run the tests
```

## Results (on my synthetic data, 5000 rows, 80/20 stratified split, thresholds tuned on train CV)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.35 | 0.77 | 0.58 | 0.64 | 0.61 | 0.81 |
| RandomForest | 0.25 | 0.68 | 0.45 | 0.77 | 0.57 | 0.77 |
| GradientBoosting | 0.25 | 0.73 | 0.51 | 0.76 | 0.61 | 0.80 |

(Your numbers will differ slightly — the data generator uses a fixed seed but the models have their own randomness.)

Two things worth noting. First, the default 0.5 decision threshold badly under-predicts churn here (only ~28% of customers churn), so I tune the threshold per model with 3-fold CV on the training set — that alone took recall from ~0.45 to ~0.76. Second, logistic regression has the best ROC-AUC but gradient boosting wins on F1, which is the metric I actually care about for churn (catching churners matters more than overall accuracy). The precision/recall tradeoff is real: at these thresholds we catch most churners but flag a fair number of false positives too.

## Project layout

```
data/generate.py   synthetic dataset generator (clearly labeled synthetic)
eda.py             exploration script -> reports/figures/
features.py        preprocessing: scaling, one-hot, stratified split (no leakage)
train.py           train + compare 3 models, save best + preprocessor
evaluate.py        classification report on held-out test set
api.py             Flask: POST /predict + a small HTML form at /
tests/             pytest for the preprocessing pipeline
```

## What I'd try next

- Hyperparameter tuning with GridSearchCV instead of default params
- Calibration curves — the probabilities matter more than the labels for churn
- SHAP values to explain individual predictions in the API response
- A real dataset (e.g. the IBM Telco churn data on Kaggle) to see how the workflow holds up
