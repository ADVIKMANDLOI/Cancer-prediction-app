import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar header
st.sidebar.title("🩺 Breast Cancer Prediction App")
st.sidebar.markdown("Predict breast cancer malignancy using XGBoost. For educational use only.")

# Main content
st.title("Welcome to Breast Cancer Prediction App")
st.markdown("""
This app uses machine learning to predict whether a breast tumor is benign or malignant based on medical features.
""")

# About Model section
st.header("About Model")
st.markdown("""
- **Model**: XGBoost Classifier
- **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)
- **Target**: Diagnosis (M = Malignant, B = Benign)
- **Accuracy**: ~98% (after hyperparameter tuning)
- **Techniques**: SMOTE oversampling, GridSearchCV
""")

# Disclaimer
st.warning("⚠️ This tool is for educational use only and not for medical diagnosis. Consult a healthcare professional for actual medical advice.")

# How It Works section
st.header("How It Works")
st.markdown("""
1. Navigate to the **Predict** page to input medical features and get predictions.
2. Check the **Insights** page for model performance metrics and visualizations.
3. All predictions are based on the trained XGBoost model.
""")

# Dataset preview (optional)
st.header("Dataset Preview")
try:
    # Use relative path from app directory to repository root
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data.csv")
    data = pd.read_csv(data_path)
    st.dataframe(data.head(5))
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}. Please ensure 'data.csv' is in the project root.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Advik Mandloi | For educational use only")
