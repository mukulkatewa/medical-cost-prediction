"""
Train and evaluate regression models for medical insurance cost prediction.

Usage:
    python src/train.py --data data/insurance.csv --model-out models/model.joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
TARGET = "charges"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.isnull().sum().sum() > 0:
        raise ValueError("Unexpected missing values in insurance dataset")
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary"), CATEGORICAL_FEATURES),
        ]
    )


def build_candidates(preprocessor: ColumnTransformer) -> dict:
    return {
        "linear_regression": Pipeline(
            [("prep", preprocessor), ("model", LinearRegression())]
        ),
        "polynomial_regression": Pipeline(
            [
                ("prep", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", LinearRegression()),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("prep", preprocessor),
                ("model", DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
                    ),
                ),
            ]
        ),
    }


def evaluate(model, x_test, y_test) -> dict:
    preds = model.predict(x_test)
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/insurance.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.data)
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    candidates = build_candidates(preprocessor)

    results = {}
    fitted = {}
    for name, pipeline in candidates.items():
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="r2")
        pipeline.fit(x_train, y_train)
        metrics = evaluate(pipeline, x_test, y_test)
        metrics["cv_r2_mean"] = float(cv_scores.mean())
        metrics["cv_r2_std"] = float(cv_scores.std())
        results[name] = metrics
        fitted[name] = pipeline
        print(f"[{name}] test_r2={metrics['r2']:.4f} rmse={metrics['rmse']:.2f} "
              f"cv_r2={metrics['cv_r2_mean']:.4f}+/-{metrics['cv_r2_std']:.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (test R2={results[best_name]['r2']:.4f})")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Saved best model to {args.model_out}")
    print(f"Saved metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
