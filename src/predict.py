"""predict.py

Simple CLI to run batch predictions using saved model/scaler and feature list.

Usage:
    python -m src.predict --input Data/Processed/cleaned_diabetic_data.csv --output predictions.csv
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='Results/Model_files/model.joblib')
    parser.add_argument('--scaler', default='Results/Model_files/scaler.joblib')
    parser.add_argument('--metrics', default='Results/Model_files/metrics.json')
    return parser.parse_args()


def main():
    args = _parse_args()
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    df = pd.read_csv(args.input)

    # attempt to load features from metrics.json
    try:
        import json

        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
            features = metrics.get('features')
    except Exception:
        features = None

    if features:
        missing = [c for c in features if c not in df.columns]
        for c in missing:
            df[c] = 0
        df = df[features]

    X = scaler.transform(df) if scaler is not None else df.values
    preds = model.predict_proba(X)[:, 1]
    df['predicted_readmit_proba'] = preds
    df.to_csv(args.output, index=False)
    print(f'Wrote predictions to {args.output}')


if __name__ == '__main__':
    main()
