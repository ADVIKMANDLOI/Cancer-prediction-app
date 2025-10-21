🩺 Breast Cancer Prediction ML Project
A comprehensive machine learning project featuring a production-ready Streamlit web application for real-time breast cancer diagnosis prediction using advanced ML algorithms.

🎯 Project Overview
This project demonstrates the complete ML pipeline: from data preprocessing and model training to web deployment. It includes three powerful algorithms (SVM, Random Forest, XGBoost) with a beautiful, interactive Streamlit app for real-world predictions.

🤖 Machine Learning Models
Support Vector Machine (SVM) - ~97% accuracy
Random Forest - ~96% accuracy
XGBoost - ~98% accuracy (Best performing model)
📊 Dataset
We use the Breast Cancer Wisconsin Dataset from UCI ML Repository, containing 30 medical features for binary classification (Malignant/Benign).

🏗️ Project Structure

Cancer-Prediction-ML-Project/
├── Cancer_prediction_model.ipynb    # Model training & evaluation
├── train_model.py                   # Model training script
├── data.csv                         # Breast cancer dataset
├── XGBoost.pkl                      # Trained XGBoost model
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── app/                             # Streamlit web application
│   ├── app.py                       # Main application file
│   └── pages/
│       ├── predict.py               # Prediction interface
│       └── insights.py              # Model analytics & visualizations
├── models/                          # Saved ML models
│   └── xgboost_model.pkl
├── outputs/                         # Model evaluation plots
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── accuracy_comparison_barplot.png
└── assets/                          # Static assets
    └── logo.png
🚀 Streamlit Web Application
Features:
🏠 Home Page: Welcome dashboard with model overview and dataset preview
🔮 Prediction Page: 30-feature medical input form with real-time XGBoost predictions
📊 Insights Page: Dynamic visualizations, model metrics, and SHAP explanations
🎨 Professional UI: Medical theme with responsive design and clean layout
Key Capabilities:
Real-time cancer prediction with confidence scores
Interactive feature importance analysis using SHAP
Dynamic correlation heatmaps and model comparisons
Smart input reset functionality
Mobile-responsive design
🛠️ Installation & Setup
Prerequisites:
Python 3.8+
pip package manager
Local Setup:

# Clone or download the project
cd Cancer-Prediction-ML-Project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
Access the App:
Open your browser to: http://localhost:8501
📈 Model Performance
| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| XGBoost | ~98% | 0.97 | 0.98 |
| SVM | ~97% | 0.96 | 0.97 |
| Random Forest | ~96% | 0.95 | 0.96 |

Evaluation Metrics:
Confusion Matrix
Classification Report
ROC-AUC Curves
Feature Importance Analysis
🔬 Technical Implementation
Data Preprocessing:
Feature scaling and normalization
SMOTE oversampling for class balance
Train/test split (80/20)
Model Training:
Hyperparameter tuning with GridSearchCV
Cross-validation for robust evaluation
Model serialization with joblib
Web Application:
Multi-page Streamlit architecture
Session state management for inputs
Caching for performance optimization
Error handling and loading states
📊 Visualizations
Correlation Heatmap: Feature relationships analysis
Model Comparison Chart: Accuracy comparison across algorithms
SHAP Feature Importance: Model explainability plots
Confusion Matrix: Prediction performance visualization

🧪 Usage Examples
Making Predictions:
Navigate to the "Predict" page
Enter 30 medical feature values
Click "Predict" for instant results
View confidence scores and explanations
Exploring Insights:
Visit the "Insights" page
Examine model performance metrics
Interact with dynamic visualizations
Analyze feature importance
🤝 Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Advik Mandloi

🙏 Acknowledgments
UCI Machine Learning Repository for the dataset
Streamlit team for the amazing web framework
Scikit-learn, XGBoost, and SHAP communities
⭐ Star this repository if you found it helpful!

🔗 Live Demo: 

📧 Contact: mr.advikmandloi@gmail.com
## Author
Advik Mandloi | [LinkedIn](https://www.linkedin.com/in/advik-mandloi)
