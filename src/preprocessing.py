"""preprocessing.py

Reusable data cleaning utilities extracted from the notebook.

Functions:
- clean_df(df): apply transformations and return cleaned df
- load_and_clean(input_path, output_path): read CSV, clean, save

Usage:
	python -m src.preprocessing --input Data/Processed/diabetic_data.csv --output Data/Processed/cleaned_diabetic_data.csv
"""

import argparse
import os
import pandas as pd
import numpy as np


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply deterministic cleaning steps to the dataset.

	This mirrors the steps in your notebook.
	"""
	# Replace placeholder missing values
	df = df.copy()
	df.replace("?", np.nan, inplace=True)

	# Drop identifiers if present
	for col in ["encounter_id", "patient_nbr"]:
		if col in df.columns:
			df.drop(columns=col, inplace=True)

	# Target simplification: consider '<30' as positive class
	if "readmitted" in df.columns:
		df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

	# Normalize gender
	if "gender" in df.columns:
		df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan).fillna("Unknown")

	# Map age buckets to numeric
	age_mapping = {
		"[0-10)": 5,
		"[10-20)": 15,
		"[20-30)": 25,
		"[30-40)": 35,
		"[40-50)": 45,
		"[50-60)": 55,
		"[60-70)": 65,
		"[70-80)": 75,
		"[80-90)": 85,
		"[90-100)": 95,
	}
	if "age" in df.columns:
		df["age"] = df["age"].replace(age_mapping)
		# convert where possible to integer
		try:
			df["age"] = df["age"].astype(int)
		except Exception:
			pass

	# Drop high-cardinality or noisy columns if present
	drop_cols = ["weight", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]
	df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

	# Encode some medication-related columns
	if "insulin" in df.columns:
		insulin_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
		df["insulin"] = df["insulin"].map(insulin_map).fillna(0)

	if "change" in df.columns:
		df["change"] = df["change"].map({"Ch": 1, "No": 0}).fillna(0)

	if "diabetesMed" in df.columns:
		df["diabetesMed"] = df["diabetesMed"].map({"Yes": 1, "No": 0}).fillna(0)

	# One-hot encode remaining object (categorical) columns
	cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
	if len(cat_cols) > 0:
		df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

	# Fill remaining numeric missing values with median
	try:
		df.fillna(df.median(numeric_only=True), inplace=True)
	except Exception:
		# fallback: fill any remaining NaNs with 0
		df.fillna(0, inplace=True)

	return df


def load_and_clean(input_path: str, output_path: str) -> None:
	"""Load CSV, clean it, and save cleaned CSV to output_path.
	Does not overwrite input_path.
	"""
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"Input file not found: {input_path}")

	df = pd.read_csv(input_path)
	print(f"Loaded {input_path} with shape {df.shape}")
	cleaned = clean_df(df)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	cleaned.to_csv(output_path, index=False)
	print(f"Saved cleaned data to {output_path} with shape {cleaned.shape}")


def _parse_args():
	parser = argparse.ArgumentParser(description="Clean diabetic dataset and write cleaned CSV")
	parser.add_argument("--input", default="Data/Processed/diabetic_data.csv", help="Input CSV path")
	parser.add_argument("--output", default="Data/Processed/cleaned_diabetic_data.csv", help="Output cleaned CSV path")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	load_and_clean(args.input, args.output)
