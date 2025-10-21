import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Sidebar header
st.sidebar.title("🩺 Breast Cancer Prediction App")
st.sidebar.markdown("Predict breast cancer malignancy using XGBoost. For educational use only.")

# Page title
st.title("Model Insights & Performance")

# Load model and data
@st.cache_resource
def load_model():
    try:
        # Use relative path from pages directory to repository root
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "xgboost_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}.")
        return None

@st.cache_data
def load_data():
    try:
        # Use relative path from pages directory to repository root
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data.csv")
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}.")
        return None

model = load_model()
data = load_data()

if model is None or data is None:
    st.stop()

# Prepare data for evaluation
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})  # Encode target

# Metrics section
st.header("Model Performance Metrics")

# Calculate metrics
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{accuracy*100:.1f}%")
with col2:
    st.metric("Precision", f"{precision*100:.1f}%")
with col3:
    st.metric("Recall", f"{recall*100:.1f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
st.write(cm)

# Visualizations section
st.header("Visualizations")

# Create columns for organized layout
col1, col2 = st.columns(2)

with col1:
    with st.expander("Feature Correlation Heatmap", expanded=True):
        try:
            # Generate correlation heatmap dynamically
            corr_matrix = X.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax, cbar=True)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")

with col2:
    with st.expander("Model Accuracy Comparison", expanded=True):
        try:
            # Sample accuracy data for different models (you can replace with actual data)
            models = ['SVM', 'Random Forest', 'XGBoost']
            accuracies = [0.95, 0.97, 0.98]  # Replace with actual accuracies

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(models, accuracies, color=['skyblue', 'lightgreen', 'coral'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim(0, 1)
            for i, v in enumerate(accuracies):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating accuracy comparison: {e}")

# Full-width SHAP plot
with st.expander("SHAP Feature Importance", expanded=True):
    try:
        # Handle GridSearchCV model
        if hasattr(model, 'best_estimator_'):
            actual_model = model.best_estimator_
        else:
            actual_model = model

        # Sample data for SHAP (using a subset for performance)
        X_sample = X.sample(min(100, len(X)), random_state=42)

        explainer = shap.TreeExplainer(actual_model)
        shap_values = explainer.shap_values(X_sample)

        # Create figure and generate SHAP plot
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar Plot)")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")
