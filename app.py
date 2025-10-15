import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

st.title("Diabetes Prediction Model")
st.write("This application predicts the likelihood of diabetes readmission based on patient data.")

# Try to load model and scaler
MODEL_PATH = Path('Results/Model_files/model.joblib')
SCALER_PATH = Path('Results/Model_files/scaler.joblib')
FEATURES_PATH = Path('Results/Model_files/metrics.json')

model = None
scaler = None
features = None
if MODEL_PATH.exists():
	model = joblib.load(MODEL_PATH)
if SCALER_PATH.exists():
	scaler = joblib.load(SCALER_PATH)
if FEATURES_PATH.exists():
	import json

	with open(FEATURES_PATH, 'r') as f:
		metrics = json.load(f)
		features = metrics.get('features')


def predict_df(df: pd.DataFrame):
	if features:
		# align columns
		missing = [c for c in features if c not in df.columns]
		for c in missing:
			df[c] = 0
		df = df[features]
	if scaler is not None:
		X = scaler.transform(df)
	else:
		X = df.values
	preds = model.predict_proba(X)[:, 1]
	return preds


st.header('Single record prediction')
if model is None:
	st.warning('No trained model found in Results/Model_files/. Train a model first (see README).')
else:
	if features is None:
		st.error('Model feature list not found. Predictions may fail.')
	else:
		# Build a simple form from the first N features (numeric ones)
		sample_features = features[:8]
		form = st.form('single')
		inputs = {}
		for feat in sample_features:
			inputs[feat] = form.number_input(feat, value=0.0)
		submit = form.form_submit_button('Predict')
		if submit:
			df_in = pd.DataFrame([inputs])
			pred = predict_df(df_in)[0]
			st.write(f'Predicted probability of readmission within 30 days: {pred:.3f}')
			# SHAP explanation for this single prediction
			if st.button('Explain prediction'):
				if 'Linear' in type(model).__name__ or hasattr(model, 'coef_'):
					explainer = shap.LinearExplainer(model, scaler.transform(pd.DataFrame([inputs])), feature_perturbation='interventional')
				else:
					explainer = shap.Explainer(model, scaler.transform(pd.DataFrame([inputs])))
				shap_values = explainer.shap_values(scaler.transform(pd.DataFrame([inputs])))
				# shap_values may be a list (for classes) — handle accordingly
				vals = shap_values if not isinstance(shap_values, list) else shap_values[1]
				# Create a bar chart
				fig, ax = plt.subplots()
				feat_names = list(df_in.columns)
				ax.barh(feat_names, vals.flatten())
				ax.set_xlabel('SHAP value')
				ax.set_title('Feature contributions')
				st.pyplot(fig)

st.header('Batch prediction (CSV upload)')
uploaded = st.file_uploader('Upload a CSV with the same cleaned features', type=['csv'])
if uploaded is not None and model is not None:
	df = pd.read_csv(uploaded)
	preds = predict_df(df)
	df['predicted_readmit_proba'] = preds
	st.dataframe(df.head())
	st.download_button('Download predictions CSV', df.to_csv(index=False), 'predictions.csv')
