import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('data.csv')
df = data
df['diagnosis'] = df['diagnosis'].astype('category')
df['diagnosis'] = df['diagnosis'].cat.codes

x = df.drop('diagnosis', axis=1).drop('id', axis=1)
y = df['diagnosis']

# Oversampling
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(x, y)

# Function to fit model
def FitModel(X, Y, algo_name, algorithm, gridSearchParams, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    grid = GridSearchCV(estimator=algorithm, param_grid=gridSearchParams,
                        cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(x_test)
    cm = confusion_matrix(y_test, pred)
    print('Best Params :', best_params)
    print('Classification Report:', classification_report(y_test, pred))
    print('Accuracy Score', accuracy_score(y_test, pred))
    print('Confusion Matrix :\n', cm)
    pickle.dump(grid_result, open(algo_name + '.pkl', 'wb'))
    return grid_result

# Train XGBoost
xgb_param = {'n_estimators': [100, 500, 1000, 2000]}
model = FitModel(X_res, Y_res, 'XGBoost', XGBClassifier(), xgb_param, cv=10)
