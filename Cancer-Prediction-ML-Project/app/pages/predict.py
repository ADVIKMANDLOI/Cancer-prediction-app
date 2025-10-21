import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Sidebar header
st.sidebar.title("🩺 Breast Cancer Prediction App")
st.sidebar.markdown("Predict breast cancer malignancy using XGBoost. For educational use only.")

# Page title
st.title("Predict Breast Cancer Diagnosis")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Use relative path from app directory to project root
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "xgboost_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Please ensure 'xgboost_model.pkl' is in the models/ directory.")
        return None

model = load_model()

if model is None:
    st.stop()

# Feature names (based on dataset)
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Input form
st.header("Input Medical Features")
st.markdown("Enter the values for the 30 features below:")

# Group inputs using columns
col1, col2, col3 = st.columns(3)

inputs = {}
for i, feature in enumerate(feature_names):
    key = f"input_{feature.replace(' ', '_').replace('.', '_')}"
    if i % 3 == 0:
        with col1:
            inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.4f", key=key)
    elif i % 3 == 1:
        with col2:
            inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.4f", key=key)
    else:
        with col3:
            inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.4f", key=key)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display result
    st.header("Prediction Result")
    if prediction == 1:  # Assuming 1 = Malignant
        st.error(f"⚠️ Malignant (Confidence: {prediction_proba[1]*100:.1f}%)")
    else:
        st.success(f"✅ Benign (Confidence: {prediction_proba[0]*100:.1f}%)")

# Reset button
if st.button("Reset Inputs"):
    # Clear all input session states
    input_keys = [f"input_{feature.replace(' ', '_').replace('.', '_')}" for feature in feature_names]
    for key in input_keys:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Model Info expander
with st.expander("Model Info"):
    st.write("**Model**: XGBoost Classifier")
    st.write("**Dataset**: Breast Cancer Wisconsin")
    st.write("**Hyperparameter tuning**: GridSearchCV")
    st.write("**Features**: 30 numerical features from medical imaging")
