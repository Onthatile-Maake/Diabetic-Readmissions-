"""train_model.py

Train a baseline model on the cleaned dataset and save artifacts.

This script:
- loads cleaned CSV
- splits train/test
- uses SMOTE to balance the training set
- trains a LogisticRegression baseline
- saves model (joblib), metrics (json), and ROC plot (png)

Usage:
	python -m src.train_model --input Data/Processed/cleaned_diabetic_data.csv \
		--model-out Results/Model_files/model.joblib --metrics-out Results/Model_files/metrics.json
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
try:
	from imblearn.over_sampling import SMOTE  # type: ignore
	_HAS_SMOTE = True
except Exception:
	SMOTE = None
	_HAS_SMOTE = False
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
							 roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(input_path: str, model_out: str, metrics_out: str, roc_out: str, scaler_out: str = None):
	df = pd.read_csv(input_path)
	if 'readmitted' not in df.columns:
		raise ValueError('Target column "readmitted" not found in cleaned dataframe')

	X = df.drop('readmitted', axis=1)
	y = df['readmitted']

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=42, stratify=y
	)

	# Scale numeric features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# record feature names for later inference
	feature_names = list(X.columns)

	if _HAS_SMOTE:
		print("SMOTE is available — applying SMOTE to scaled training data")
		smote = SMOTE(random_state=42)
		X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
		model = LogisticRegression(max_iter=5000, random_state=42)
		model.fit(X_train_res, y_train_res)
	else:
		print(
			"SMOTE not available or incompatible — falling back to class_weight='balanced' on LogisticRegression (scaled features)"
		)
		model = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced')
		model.fit(X_train_scaled, y_train)

	# Use the scaled test set for predictions
	y_pred = model.predict(X_test_scaled)
	y_proba = model.predict_proba(X_test_scaled)[:, 1]

	metrics = {
		'accuracy': float(accuracy_score(y_test, y_pred)),
		'roc_auc': float(roc_auc_score(y_test, y_proba)),
		'classification_report': classification_report(y_test, y_pred, output_dict=True)
	}

	# Ensure output directories exist
	Path(model_out).parent.mkdir(parents=True, exist_ok=True)
	Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
	Path(roc_out).parent.mkdir(parents=True, exist_ok=True)
	if scaler_out:
		Path(scaler_out).parent.mkdir(parents=True, exist_ok=True)

	# Save model and metrics
	joblib.dump(model, model_out)
	# Save scaler if requested
	if scaler_out:
		joblib.dump(scaler, scaler_out)
	# include feature names in metrics for inference
	metrics['features'] = feature_names
	with open(metrics_out, 'w') as f:
		json.dump(metrics, f, indent=2)

	# ROC plot
	fpr, tpr, _ = roc_curve(y_test, y_proba)
	plt.figure()
	plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
	plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc='lower right')
	plt.savefig(roc_out)
	plt.close()

	print(f"Model saved to {model_out}")
	print(f"Metrics saved to {metrics_out}")
	print(f"ROC plot saved to {roc_out}")


def _parse_args():
	parser = argparse.ArgumentParser(description='Train a baseline model and save artifacts')
	parser.add_argument('--input', default='Data/Processed/cleaned_diabetic_data.csv')
	parser.add_argument('--model-out', default='Results/Model_files/model.joblib')
	parser.add_argument('--metrics-out', default='Results/Model_files/metrics.json')
	parser.add_argument('--roc-out', default='Results/Visuals_files/roc.png')
	parser.add_argument('--scaler-out', default=None, help='Optional path to save fitted scaler (joblib)')
	return parser.parse_args()


if __name__ == '__main__':
	args = _parse_args()
	train_and_evaluate(args.input, args.model_out, args.metrics_out, args.roc_out)

