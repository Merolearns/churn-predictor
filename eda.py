"""Exploratory analysis for the synthetic churn dataset.

Just prints the basics and saves a few plots to reports/figures/:
class balance, churn rate by contract type, churn by tenure bucket,
and a correlation heatmap for the numeric columns.

Run: python eda.py [--data data/churn.csv]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "reports/figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(args.data)
    print(f"rows: {len(df)}, cols: {len(df.columns)}")
    print(f"\nchurn rate: {df['churn'].mean():.1%}")
    print("\nmissing values:\n", df.isna().sum().to_string())

    # 1. churn by contract type
    by_contract = df.groupby("contract_type")["churn"].mean().sort_values()
    print("\nchurn rate by contract:\n", by_contract.to_string())
    by_contract.plot.barh(title="Churn rate by contract type")
    plt.xlabel("churn rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_contract.png")
    plt.close()

    # 2. churn by tenure bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49+"]
    )
    by_tenure = df.groupby("tenure_bucket", observed=True)["churn"].mean()
    print("\nchurn rate by tenure:\n", by_tenure.to_string())
    by_tenure.plot.bar(title="Churn rate by tenure (months)")
    plt.ylabel("churn rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_tenure.png")
    plt.close()

    # 3. numeric correlations
    num_cols = ["tenure", "monthly_charges", "total_charges", "support_calls", "churn"]
    corr = df[num_cols].corr()
    print("\ncorrelation with churn:\n", corr["churn"].sort_values().to_string())

    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(num_cols)), num_cols)
    ax.set_title("Correlation heatmap (numeric cols)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/correlation.png")
    plt.close(fig)

    # 4. support calls distribution by churn
    df.groupby("support_calls")["churn"].mean().plot(
        marker="o", title="Churn rate vs support calls"
    )
    plt.xlabel("support calls")
    plt.ylabel("churn rate")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/churn_by_support_calls.png")
    plt.close()

    print(f"\nfigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
