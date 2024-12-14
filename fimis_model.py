import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Step 1: Load and Explore the Dataset
data = pd.read_csv('Churn_Modelling.csv')

# Separating features (X) and the target variable (y)
X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = data['Exited']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Preprocessing the Data
categorical_cols = ['Geography', 'Gender']
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('HasCrCard')

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

# Define a pipeline that includes both preprocessing and model training for the ensemble
ensemble_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
        ('xgb', XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)),
        ('log_reg', LogisticRegression(class_weight='balanced', random_state=42))
    ], voting='soft'))
])

# Train the ensemble model using the pipeline
ensemble_pipeline.fit(X_train, y_train)

# Save the ensemble model
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_pipeline, f)

# Step 3: Preprocess the training and test sets for individual models
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Step 4: Train and Evaluate Individual Models

# Random Forest Classifier
rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_clf.fit(X_train_transformed, y_train)
rf_pred = rf_clf.predict(X_test_transformed)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc:.2f}")
print(classification_report(y_test, rf_pred))

# XGBoost Classifier
xgb_clf = XGBClassifier(scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]), random_state=42)
xgb_clf.fit(X_train_transformed, y_train)
xgb_pred = xgb_clf.predict(X_test_transformed)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.2f}")
print(classification_report(y_test, xgb_pred))

# Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train_transformed, y_train)
log_reg_pred = log_reg.predict(X_test_transformed)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
print(f"Logistic Regression Accuracy: {log_reg_acc:.2f}")
print(classification_report(y_test, log_reg_pred))

# Step 5: Evaluate the Ensemble Model
ensemble_pred = ensemble_pipeline.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Model Accuracy: {ensemble_acc:.2f}")
print(classification_report(y_test, ensemble_pred))

# Step 6: Plot ROC Curves and AUC Scores

# Get predicted probabilities for ROC curve
rf_probs = rf_clf.predict_proba(X_test_transformed)[:, 1]
xgb_probs = xgb_clf.predict_proba(X_test_transformed)[:, 1]
log_reg_probs = log_reg.predict_proba(X_test_transformed)[:, 1]
ensemble_probs = ensemble_pipeline.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC for each model
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg_probs)
roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

# Plot ROC Curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, color='orange', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_log_reg, tpr_log_reg, color='green', label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
plt.plot(fpr_ensemble, tpr_ensemble, color='purple', label=f'Ensemble (AUC = {roc_auc_ensemble:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Step 7: Visualize Model AUC Scores in a Bar Chart
models = ['Random Forest', 'XGBoost', 'Logistic Regression', 'Ensemble']
auc_values = [roc_auc_rf, roc_auc_xgb, roc_auc_log_reg, roc_auc_ensemble]

plt.figure(figsize=(8, 6))
plt.bar(models, auc_values, color=['blue', 'orange', 'green', 'purple'])
plt.xlabel('Model')
plt.ylabel('AUC Score')
plt.title('AUC Scores for Different Models')
plt.ylim(0.5, 1.0)
plt.show()


# Plotting feature importances for Random Forest
importances = rf_clf.feature_importances_  # Correct variable name

# Get the feature names after preprocessing
encoder = preprocessor.named_transformers_['cat']  # Access the one-hot encoder
encoded_cat_features = encoder.get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numeric_cols, encoded_cat_features])

# Sort feature importances
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Random Forest)')
plt.bar(range(X_train_transformed.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train_transformed.shape[1]), all_feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()


import joblib

# Save individual models
joblib.dump(rf_clf, 'random_forest_model.pkl')
joblib.dump(xgb_clf, 'xgboost_model.pkl')
joblib.dump(log_reg, 'logistic_regression_model.pkl')

# Save the ensemble model (pipeline)
joblib.dump(ensemble_pipeline, 'ensemble_model.pkl')

joblib.dump(preprocessor, 'preprocessor.pkl')
