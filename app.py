from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained models
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')
log_reg_model = joblib.load('logistic_regression_model.pkl')
ensemble_model = joblib.load('ensemble_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

app = Flask(__name__)

# Define a route for the chatbot
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Data sent by user
    df = pd.DataFrame(data, index=[0])  # Convert to DataFrame

    # Preprocess the input data
    X_transformed = preprocessor.transform(df)

    # Get predictions from each model
    rf_pred = rf_model.predict(X_transformed)
    xgb_pred = xgb_model.predict(X_transformed)
    log_reg_pred = log_reg_model.predict(X_transformed)
    ensemble_pred = ensemble_model.predict(X_transformed)

    # Return the predictions as a JSON response
    return jsonify({
        'Random Forest Prediction': int(rf_pred[0]),
        'XGBoost Prediction': int(xgb_pred[0]),
        'Logistic Regression Prediction': int(log_reg_pred[0]),
        'Ensemble Model Prediction': int(ensemble_pred[0])
    })

if __name__ == '__main__':
    app.run(debug=True)
