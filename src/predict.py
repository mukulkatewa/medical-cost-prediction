"""
Predict medical insurance charges for a single applicant using a trained model.

Usage:
    python src/predict.py --model models/model.joblib --age 29 --sex male \
        --bmi 27.3 --children 1 --smoker no --region southeast
"""
import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--bmi", type=float, required=True)
    parser.add_argument("--children", type=int, required=True)
    parser.add_argument("--smoker", choices=["yes", "no"], required=True)
    parser.add_argument(
        "--region",
        choices=["southwest", "southeast", "northwest", "northeast"],
        required=True,
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    row = pd.DataFrame(
        [
            {
                "age": args.age,
                "bmi": args.bmi,
                "children": args.children,
                "sex": args.sex,
                "smoker": args.smoker,
                "region": args.region,
            }
        ]
    )
    prediction = model.predict(row)[0]
    print(f"Predicted medical charges: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
