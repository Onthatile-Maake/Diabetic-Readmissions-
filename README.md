# 30-Day Diabetes Readmission Prediction

## Overview

The 30-Day Diabetes Readmission Prediction project aims to analyze healthcare data to predict the likelihood of hospital readmission for diabetic patients within 30 days of discharge. This analysis helps healthcare providers take proactive measures to improve patient outcomes.

## Dataset

The project utilizes the dataset `diabetes_data.csv`.The dataset  contains anonymized  information about  patients, including their demographics, medical history, and readmission status.

## Files

- `diabetes_data.csv`: The dataset used for analysis is found in the Data folder.
- `eda_cleaning.ipynb`: Jupyter notebook for exploratory data analysis and data cleaning.
- `streamlit_app.py`: The main application file that runs the predictive model and serves the user interface using Streamlit.
- `Best_model_XGBoost_Training.ipynb`: Jupyter notebook for training and saving the best XGBoost model.
- `Logistic_Regression_Model.ipynb`: Jupyter notebook for training a logistic regression model.
- `Models_training_and_comparison.ipynb`: Notebook for training and comparing multiple models.
- `Random_forest_model_training.ipynb`: Jupyter notebook for training a Random Forest model.
- `XGBoost_Model.ipynb`: Jupyter notebook specifically for training the XGBoost model.
- `model_evaluation.ipynb`: Notebook for evaluating model performance.
- `feature_info.json`: A JSON file containing information about features used in the model, including their types and descriptions.
- `requirements.txt`: Contains the necessary Python packages for running the application.
- `environment.yml`: YAML file for setting up the project environment using Conda.
- `README.md`: This documentation file providing an overview of the project.

## Features

- The application provides insights into patient data and predictions regarding readmission risks.
- Users can explore visualizations that highlight key factors contributing to readmissions.
- The dashboard allows users to input patient data and receive immediate predictions.
- The application includes tabs for various analyses and visualizations.

## Usage

To run the application, follow these steps:

1. Install the required packages:
   ```bash
   -r pip install requirements.txt
2. Alternatively, create a Conda environment using:
   ```bash
   conda env create -f environment.yml
   
3. Run the application:
   ```bash
   streamlit run streamlit_app.py

## Model Performance

The predictive models are evaluated based on metrics such as accuracy, precision, recall, and AUC-ROC score. 
The current model (xgboost) used in the prediction app achieves a reasonable performance, but there is potential for further improvements through hyperparameter 
tuning and feature engineering.

## License

# Academic License

This project is licensed under the Academic License. You are free to use, modify, and distribute the project for educational purposes, provided that you give  
appropriate credit to the original authors.

   pip install -r requirements.txt
