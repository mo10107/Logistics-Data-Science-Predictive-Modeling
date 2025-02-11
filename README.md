# 🚛 Logistics Data Science & Predictive Modeling  

## 📌 Project Overview  
This project focuses on **logistics data analysis and predictive modeling** to **forecast delivery times with high accuracy**.  
It integrates **data preprocessing, feature engineering, advanced machine learning models, ensemble learning, and model deployment**.  

## 🎯 Key Features  
✅ **Data Integration**: Combines multiple logistics datasets (`processed_logistics_data.csv`, `pca_results.csv`, `cleaned_and_engineered_logistics_data.csv`).  
✅ **Feature Engineering**: Applies **Lasso Regularization, Recursive Feature Elimination (RFE)** for optimal feature selection.  
✅ **Advanced ML Models**: Trains **Linear Regression, XGBoost, LightGBM, and Artificial Neural Networks (ANNs)**.  
✅ **Ensemble Learning**: Implements **Model Averaging & Stacking Regressor** to boost prediction accuracy.  
✅ **Cross-Validation & Performance Evaluation**: Uses **5-Fold Cross-Validation**, **MAE, RMSE, and R² scores**.  
✅ **Deployment-Ready**: Saves the best model (`best_model.pkl`) and provides an **inference pipeline for real-world predictions**.  

## 🛠️ Installation  
Ensure Python 3.8+ is installed, then run:  
```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow joblib

📂 Dataset
The model is trained on logistics datasets, combining 135 features to predict delivery times.
Key datasets:

processed_logistics_data.csv – Cleaned logistics records.
pca_results.csv – Dimensionality-reduced features.
cleaned_and_engineered_logistics_data.csv – Feature-engineered dataset.

🏗️ Project Workflow
1️⃣ Data Preprocessing & Integration
Merges 3 datasets to create a 135-feature matrix.
Identifies actual_time as the target variable.
Handles missing values & duplicate rows.
2️⃣ Train-Test Splitting
Splits data into 80% training (181 samples) & 20% testing (46 samples).
Saves split datasets for reproducibility.
3️⃣ Baseline Modeling
Implements Linear Regression as a benchmark model.
Achieves MAE: 0.0019, RMSE: 0.0087, R²: 0.9998.
4️⃣ Advanced Modeling & Hyperparameter Tuning
Uses Lasso Regularization & Recursive Feature Elimination (RFE) for feature selection.
Trains ANN (50 epochs), XGBoost, and LightGBM, with hyperparameter tuning via Randomized Search.
Best parameters found:
XGBoost: max_depth=7, n_estimators=100, learning_rate=0.1
LightGBM: num_leaves=31, n_estimators=200, learning_rate=0.1
5️⃣ Cross-Validation & Performance Evaluation
Performs 5-Fold Cross-Validation to ensure robustness.
Achieves Mean CV R²: 0.9999 for high prediction accuracy.
6️⃣ Ensemble Learning
Model Averaging (Linear Regression, Ridge, Random Forest) → MAE: 0.0091, R²: 0.9993.
Stacking Regressor (Best Model) → MAE: 0.0021, RMSE: 0.0087, R²: 0.9998.
7️⃣ Model Deployment & Inference
Saves best model as best_model.pkl for real-time predictions.
Creates an inference pipeline for seamless integration.

🔍 Example Usage with Advanced Algorithms
1️⃣ Load & Preprocess Data

import pandas as pd

# Load datasets
processed_data = pd.read_csv("processed_logistics_data.csv")
pca_results = pd.read_csv("pca_results.csv")
cleaned_data = pd.read_csv("cleaned_and_engineered_logistics_data.csv")

# Merge datasets
X = pd.concat([processed_data, pca_results, cleaned_data], axis=1)
y = processed_data["actual_time"]

# Check dataset shape
print(f"Feature Matrix Shape: {X.shape}, Target Shape: {y.shape}")
2️⃣ Train-Test Splitting

from sklearn.model_selection import train_test_split

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train Size: {X_train.shape}, Test Size: {X_test.shape}")
3️⃣ Train XGBoost Model

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize XGBoost model with best parameters
xgb_model = xgb.XGBRegressor(max_depth=7, n_estimators=100, learning_rate=0.1)

# Train model
xgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate model performance
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Performance:\nMAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R²: {r2_xgb:.4f}")
4️⃣ Train LightGBM Model

import lightgbm as lgb

# Initialize LightGBM model with best parameters
lgb_model = lgb.LGBMRegressor(num_leaves=31, n_estimators=200, learning_rate=0.1)

# Train model
lgb_model.fit(X_train, y_train)

# Predict on test data
y_pred_lgb = lgb_model.predict(X_test)

# Evaluate model performance
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = mean_squared_error(y_test, y_pred_lgb, squared=False)
r2_lgb = r2_score(y_test, y_pred_lgb)

print(f"LightGBM Performance:\nMAE: {mae_lgb:.4f}, RMSE: {rmse_lgb:.4f}, R²: {r2_lgb:.4f}")
5️⃣ Train Artificial Neural Network (ANN)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build ANN Model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Predict on test data
y_pred_ann = model.predict(X_test)

# Evaluate performance
mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = mean_squared_error(y_test, y_pred_ann, squared=False)
r2_ann = r2_score(y_test, y_pred_ann)

print(f"ANN Performance:\nMAE: {mae_ann:.4f}, RMSE: {rmse_ann:.4f}, R²: {r2_ann:.4f}")
6️⃣ Compare Model Performances
import pandas as pd

# Create a dataframe to compare models
performance_df = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM", "ANN"],
    "MAE": [mae_xgb, mae_lgb, mae_ann],
    "RMSE": [rmse_xgb, rmse_lgb, rmse_ann],
    "R² Score": [r2_xgb, r2_lgb, r2_ann]
})

# Print results
print("\nModel Performance Comparison:\n", performance_df)
7️⃣ Save & Deploy the Best Model

import joblib
# Save the best-performing model
best_model = xgb_model  # Example: XGBoost performed best
joblib.dump(best_model, "best_model.pkl")

print("Best model saved as 'best_model.pkl'.")
8️⃣ Load & Use the Saved Model for Inference
# Load the saved model
loaded_model = joblib.load("best_model.pkl")

# Make a prediction on new data
sample_input = X_test.iloc[0].values.reshape(1, -1)
predicted_time = loaded_model.predict(sample_input)

print("Predicted Delivery Time:", predicted_time)

📊 Visualization & Analysis
Feature Importance → Identifies the most impactful predictors.
Residual Plots → Visualizes prediction errors.
Learning Curves → Analyzes model generalization.

📌 Results
Final Model: Stacking Regressor → MAE: 0.0021, R²: 0.9998.
Robust performance across cross-validation tests.
Deployment-ready model with an inference pipeline.

📜 References
Scikit-Learn Documentation
XGBoost
LightGBM

📬 Contact
For questions or contributions, reach out at moab10107@gmail.com.
