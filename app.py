import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load data and models
@st.cache_resource
def load_model():
    try:
        return joblib.load('Model/diabetic_readmission_model.joblib')
    except FileNotFoundError:
        st.error("Model file not found.")
        return None

@st.cache_data
def load_feature_info():
    try:
        return joblib.load('Model/feature_info.joblib')
    except:
        return None

@st.cache_data
def load_data():
    try:
        data_path = 'Data/Processed/cleaned_diabetic_data.csv'
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None
    except:
        return None

@st.cache_data
def load_mappings():
    try:
        mappings_path = 'Data/Processed/IDs_mapping.csv'
        if not os.path.exists(mappings_path):
            return get_default_mappings()
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        admission_mapping, discharge_mapping, source_mapping = {}, {}, {}
        
        sections = content.split('\n\n')
        for section in sections:
            lines = [line.strip() for line in section.split('\n') if line.strip()]
            if not lines:
                continue
                
            header = lines[0]
            if 'admission_type_id' in header:
                current_mapping = admission_mapping
            elif 'discharge_disposition_id' in header:
                current_mapping = discharge_mapping
            elif 'admission_source_id' in header:
                current_mapping = source_mapping
            else:
                continue
            
            for line in lines[1:]:
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(' ', 1)
                
                if len(parts) >= 2:
                    try:
                        code = int(parts[0].strip())
                        description = parts[1].strip()
                        current_mapping[code] = description
                    except ValueError:
                        continue
        
        return admission_mapping, discharge_mapping, source_mapping
        
    except Exception:
        return get_default_mappings()

def get_default_mappings():
    admission_mapping = {
        1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 
        5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped"
    }
    
    discharge_mapping = {
        1: "Discharged to home", 2: "Discharged/transferred to another short term hospital", 
        3: "Discharged/transferred to SNF", 6: "Discharged/transferred to home with home health service", 
        7: "Left AMA", 11: "Expired", 13: "Hospice / home", 14: "Hospice / medical facility"
    }
    
    source_mapping = {
        1: "Physician Referral", 2: "Clinic Referral", 7: "Emergency Room", 
        4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility (SNF)"
    }
    
    return admission_mapping, discharge_mapping, source_mapping

def main():
    st.title("🏥 Diabetes Readmission Predictor")
    st.write("Predict 30-day readmission risk for diabetes patients")
    
    # Load resources
    model = load_model()
    feature_info = load_feature_info()
    data = load_data()
    admission_mapping, discharge_mapping, source_mapping = load_mappings()
    
    if model is not None:
        st.sidebar.success("✅ Model loaded successfully!")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate", 
            ["Home", "Make Prediction", "Visualizations", "ID Mappings", "Model Info"]
        )
        
        if page == "Home":
            show_home()
        elif page == "Make Prediction":
            make_prediction(model, admission_mapping, discharge_mapping, source_mapping)
        elif page == "Visualizations":
            show_visualizations(data)
        elif page == "ID Mappings":
            show_mappings(admission_mapping, discharge_mapping, source_mapping)
        elif page == "Model Info":
            show_model_info(model, feature_info)

def show_home():
    st.header("Welcome to the Diabetes Readmission Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        This AI-powered tool helps healthcare providers predict the likelihood of diabetes patients 
        being readmitted to the hospital within 30 days.
        
        **Key Features:**
        - 🎯 **Accurate Predictions**: Machine learning model trained on clinical data
        - 📊 **Data Insights**: Visualize patterns and trends in patient data
        - 🏥 **Clinical Integration**: Uses standard medical codes and terminology
        - ⚡ **Real-time Results**: Instant risk assessment
        
        **How to use:**
        1. Go to the **Make Prediction** page
        2. Enter the patient's clinical information
        3. Click **Predict** to get the readmission risk assessment
        """)
    
    with col2:
        st.info("""
        **Quick Stats**
        - 30-day readmission prediction
        - Comprehensive risk factors
        - Evidence-based recommendations
        """)

def make_prediction(model, admission_mapping, discharge_mapping, source_mapping):
    st.header("🎯 Make a Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Demographics")
            admission_type = st.selectbox(
                "Admission Type", 
                options=list(admission_mapping.keys()), 
                format_func=lambda x: f"{x} - {admission_mapping[x]}"
            )
            discharge_disposition = st.selectbox(
                "Discharge Disposition", 
                options=list(discharge_mapping.keys()), 
                format_func=lambda x: f"{x} - {discharge_mapping[x]}"
            )
            admission_source = st.selectbox(
                "Admission Source",
                options=list(source_mapping.keys()),
                format_func=lambda x: f"{x} - {source_mapping[x]}"
            )
            age = st.selectbox("Age Group", 
                ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
            )
        
        with col2:
            st.subheader("Medical Information")
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
            num_lab_procedures = st.number_input("Number of Lab Procedures", 1, 200, 45)
            num_medications = st.number_input("Number of Medications", 1, 80, 15)
            num_procedures = st.number_input("Number of Procedures", 0, 10, 0)
            number_diagnoses = st.number_input("Number of Diagnoses", 1, 20, 9)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Diabetes Management")
            diabetes_med = st.selectbox("Diabetes Medication", ["No", "Yes"])
            insulin = st.selectbox("Insulin", ["No", "Yes"])
            max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        
        with col4:
            st.subheader("Test Results")
            change = st.selectbox("Change in Medications", ["No", "Yes"])
            A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
        
        submitted = st.form_submit_button("Predict Readmission Risk", type="primary")
        
        if submitted:
            input_data = create_input_data(
                time_in_hospital, num_lab_procedures, num_medications, num_procedures,
                number_diagnoses, admission_type, discharge_disposition, admission_source,
                change, diabetes_med, A1Cresult, max_glu_serum, insulin
            )
            
            make_prediction_with_data(model, input_data)

def create_input_data(time_in_hospital, num_lab_procedures, num_medications, num_procedures,
                     number_diagnoses, admission_type, discharge_disposition, admission_source,
                     change, diabetes_med, A1Cresult, max_glu_serum, insulin):
    
    input_data = {
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'num_procedures': num_procedures,
        'number_diagnoses': number_diagnoses,
        'admission_type_id': admission_type,
        'discharge_disposition_id': discharge_disposition,
        'admission_source_id': admission_source,
        'change': 1 if change == "Yes" else 0,
        'diabetesMed': 1 if diabetes_med == "Yes" else 0,
    }
    
    # One-hot encoded features
    input_data.update({
        'A1Cresult_None': 1 if A1Cresult == "None" else 0,
        'A1Cresult_Norm': 1 if A1Cresult == "Norm" else 0,
        'A1Cresult_>7': 1 if A1Cresult == ">7" else 0,
        'A1Cresult_>8': 1 if A1Cresult == ">8" else 0,
        'max_glu_serum_None': 1 if max_glu_serum == "None" else 0,
        'max_glu_serum_Norm': 1 if max_glu_serum == "Norm" else 0,
        'max_glu_serum_>200': 1 if max_glu_serum == ">200" else 0,
        'max_glu_serum_>300': 1 if max_glu_serum == ">300" else 0,
        'insulin_No': 1 if insulin == "No" else 0,
        'insulin_Steady': 1 if insulin == "Yes" else 0,
    })
    
    return input_data

def make_prediction_with_data(model, input_data):
    input_df = pd.DataFrame([input_data])
    
    if hasattr(model, 'feature_names_in_'):
        for feature in model.feature_names_in_:
            if feature not in input_df.columns:
                input_df[feature] = 0
        input_df = input_df[model.feature_names_in_]
    
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        st.subheader("📊 Prediction Results")
        
        if prediction[0] == 1:
            st.error("🚨 HIGH RISK OF 30-DAY READMISSION")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Risk Probability", f"{probability[0][1]:.1%}")
            with col2:
                st.write("**Clinical Recommendations:**")
                st.write("• Enhanced discharge planning")
                st.write("• Follow-up within 7 days")
                st.write("• Medication reconciliation review")
                st.write("• Care coordination with PCP")
        else:
            st.success("✅ LOW RISK OF 30-DAY READMISSION")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Risk Probability", f"{probability[0][0]:.1%}")
            with col2:
                st.write("**Recommendations:**")
                st.write("• Standard discharge process")
                st.write("• Follow-up within 14-30 days")
                st.write("• Diabetes education materials")
                
    except Exception as e:
        st.error(f"Prediction error: {e}")

def show_visualizations(data):
    st.header("📊 Data Visualizations")
    
    if data is None:
        st.warning("No data available for visualizations. Please check if 'cleaned_diabetic_data.csv' exists in Data/Processed/")
        return
    
    tab1, tab2, tab3 = st.tabs(["Patient Demographics", "Medical Features", "Readmission Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            if 'age' in data.columns:
                age_counts = data['age'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(10, 6))
                age_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Patient Age Distribution')
                ax.set_xlabel('Age Group')
                ax.set_ylabel('Number of Patients')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info("Age data not available")
        
        with col2:
            st.subheader("Gender Distribution")
            if 'gender' in data.columns:
                gender_counts = data['gender'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                gender_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title('Patient Gender Distribution')
                st.pyplot(fig)
            else:
                st.info("Gender data not available")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Time in Hospital")
            if 'time_in_hospital' in data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                data['time_in_hospital'].hist(bins=20, ax=ax, color='lightgreen')
                ax.set_title('Distribution of Hospital Stay Duration')
                ax.set_xlabel('Days in Hospital')
                ax.set_ylabel('Number of Patients')
                st.pyplot(fig)
            else:
                st.info("Time in hospital data not available")
        
        with col2:
            st.subheader("Number of Medications")
            if 'num_medications' in data.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                data['num_medications'].hist(bins=20, ax=ax, color='orange')
                ax.set_title('Distribution of Medications')
                ax.set_xlabel('Number of Medications')
                ax.set_ylabel('Number of Patients')
                st.pyplot(fig)
            else:
                st.info("Medication data not available")
    
    with tab3:
        st.subheader("Readmission Rates")
        
        # Check if readmission column exists (might have different names)
        readmission_cols = [col for col in data.columns if 'readmit' in col.lower()]
        
        if readmission_cols:
            readmission_col = readmission_cols[0]
            readmission_counts = data[readmission_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                readmission_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
                ax.set_title('Readmission Distribution')
                ax.set_xlabel('Readmission Status')
                ax.set_ylabel('Number of Patients')
                st.pyplot(fig)
            
            with col2:
                total_patients = len(data)
                readmitted_patients = readmission_counts.get(1, 0)
                readmission_rate = (readmitted_patients / total_patients) * 100
                
                st.metric("Total Patients", total_patients)
                st.metric("Readmitted Patients", readmitted_patients)
                st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
        else:
            st.info("Readmission data not available")

def show_mappings(admission_mapping, discharge_mapping, source_mapping):
    st.header("📋 ID Mappings Reference")
    
    tab1, tab2, tab3 = st.tabs(["Admission Types", "Discharge Dispositions", "Admission Sources"])
    
    with tab1:
        st.subheader("Admission Types")
        for code in sorted(admission_mapping.keys()):
            st.write(f"**{code}**: {admission_mapping[code]}")
    
    with tab2:
        st.subheader("Discharge Dispositions")
        for code in sorted(discharge_mapping.keys()):
            st.write(f"**{code}**: {discharge_mapping[code]}")
    
    with tab3:
        st.subheader("Admission Sources")
        for code in sorted(source_mapping.keys()):
            st.write(f"**{code}**: {source_mapping[code]}")

def show_model_info(model, feature_info):
    st.header("🔧 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Model Type**: {type(model).__name__}")
        if hasattr(model, 'feature_names_in_'):
            st.write(f"**Number of Features**: {len(model.feature_names_in_)}")
        
        if feature_info:
            st.write(f"**Training Date**: {feature_info.get('training_date', 'N/A')}")
            st.write(f"**Model Version**: {feature_info.get('model_type', 'N/A')}")
    
    with col2:
        st.subheader("Feature Information")
        if feature_info and 'feature_names' in feature_info:
            st.write("Top 10 Features:")
            for feature in feature_info['feature_names'][:10]:
                st.write(f"• {feature}")
        else:
            st.info("Feature information not available")
    
    if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Top 10 Most Important Features')
        st.pyplot(fig)

if __name__ == "__main__":
    main()