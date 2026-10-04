"""Generate a synthetic telecom customer churn dataset.

Everything here is made up - the point is to have a realistic-looking
dataset to practice the full ML workflow on. Real churn data is hard to
find publicly, so I generate my own with a plausible signal baked in:

- short tenure + month-to-month contract -> more likely to churn
- high monthly charges + fiber internet -> more likely to churn
- lots of support calls -> more likely to churn
- everything else is noise

Run: python data/generate.py [--n 5000] [--out data/churn.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{i:05d}" for i in range(n)]
    tenure = rng.integers(1, 73, size=n)  # months with the company
    monthly_charges = np.round(rng.uniform(20, 120, size=n), 2)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n,
        p=[0.55, 0.25, 0.20],
    )
    internet = rng.choice(
        ["DSL", "Fiber optic", "No internet"],
        size=n,
        p=[0.35, 0.45, 0.20],
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n,
        p=[0.35, 0.20, 0.20, 0.25],
    )
    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice([0, 1], size=n, p=[0.52, 0.48])
    dependents = rng.choice([0, 1], size=n, p=[0.70, 0.30])
    support_calls = rng.poisson(1.2, size=n).clip(0, 10)
    # total charges roughly = tenure * monthly, with some noise
    total_charges = np.round(
        tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n), 2
    )

    # churn signal - logistic-ish combination of the real drivers
    logit = (
        -2.2
        - 0.055 * tenure
        + 0.022 * monthly_charges
        + 1.3 * (contract == "Month-to-month")
        + 0.5 * (internet == "Fiber optic")
        + 0.4 * support_calls
        + 0.3 * senior
        - 0.3 * partner
        + rng.normal(0, 0.5, size=n)  # noise so it's not perfectly separable
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--out", default="data/churn.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = generate(n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"wrote {len(df)} rows -> {args.out}")
    print(f"churn rate: {df['churn'].mean():.1%}")
    print(df["contract_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
