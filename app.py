"""
TASK 7, 8, 9: Flask Web Application for Loan Default Prediction
===============================================================
This file contains:
- Task 7: Flask Project Setup
- Task 8: Front End (HTML Templates)
- Task 9: Backend (Handle user requests)
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessors
MODEL_LOAD_ERROR = None
try:
    # Prefer artifacts exported by notebook deployment cell.
    model_path = 'loan_model.pkl' if os.path.exists('loan_model.pkl') else 'model.pkl'
    features_path = 'features.pkl' if os.path.exists('features.pkl') else 'feature_names.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load(features_path)

    print(f"✅ Model loaded from: {model_path}")
    print("✅ Scaler and features loaded successfully!")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"⚠️ Model artifacts could not be loaded: {MODEL_LOAD_ERROR}")
    model = None
    scaler = None
    feature_names = None


def preprocess_input(input_data):
    """Match the notebook preprocessing used during training."""
    numerical_to_scale = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
    ]
    multi_cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']

    # Match binary label encoding used in training (No->0, Yes->1)
    binary_map = {'No': 0, 'Yes': 1}
    input_data['HasMortgage'] = input_data['HasMortgage'].map(binary_map).fillna(0).astype(int)
    input_data['HasDependents'] = input_data['HasDependents'].map(binary_map).fillna(0).astype(int)
    input_data['HasCoSigner'] = input_data['HasCoSigner'].map(binary_map).fillna(0).astype(int)

    # Match one-hot encoding behavior used in training.
    input_encoded = pd.get_dummies(input_data, columns=multi_cat_cols, drop_first=True)

    # Align to trained feature order and fill missing one-hot columns.
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale only the original numerical columns, like training.
    input_aligned[numerical_to_scale] = scaler.transform(input_aligned[numerical_to_scale])
    return input_aligned

# Define categorical options for the form
EDUCATION_OPTIONS = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT_OPTIONS = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL_OPTIONS = ["Single", "Married", "Divorced"]
LOAN_PURPOSE_OPTIONS = ["Home", "Auto", "Education", "Business", "Other"]
YES_NO_OPTIONS = ["Yes", "No"]


# =============================================================================
# TASK 9: Backend Routes
# =============================================================================

@app.route('/')
def home():
    """Home page with the prediction form"""
    return render_template('index.html',
                         education_options=EDUCATION_OPTIONS,
                         employment_options=EMPLOYMENT_OPTIONS,
                         marital_options=MARITAL_OPTIONS,
                         loan_purpose_options=LOAN_PURPOSE_OPTIONS,
                         yes_no_options=YES_NO_OPTIONS)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None or scaler is None or feature_names is None:
            detail = f" Details: {MODEL_LOAD_ERROR}" if MODEL_LOAD_ERROR else ''
            return render_template('error.html', error_message='Model artifacts are unavailable or incompatible on server.' + detail)

        # Get form data
        age = int(request.form['age'])
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = int(request.form['credit_score'])
        months_employed = int(request.form['months_employed'])
        num_credit_lines = int(request.form['num_credit_lines'])
        interest_rate = float(request.form['interest_rate'])
        loan_term = int(request.form['loan_term'])
        dti_ratio = float(request.form['dti_ratio'])
        education = request.form['education']
        employment_type = request.form['employment_type']
        marital_status = request.form['marital_status']
        has_mortgage = request.form['has_mortgage']
        has_dependents = request.form['has_dependents']
        loan_purpose = request.form['loan_purpose']
        has_cosigner = request.form['has_cosigner']
        
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'MonthsEmployed': [months_employed],
            'NumCreditLines': [num_credit_lines],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'DTIRatio': [dti_ratio],
            'Education': [education],
            'EmploymentType': [employment_type],
            'MaritalStatus': [marital_status],
            'HasMortgage': [has_mortgage],
            'HasDependents': [has_dependents],
            'LoanPurpose': [loan_purpose],
            'HasCoSigner': [has_cosigner]
        })

        input_final = preprocess_input(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_final)[0])
        probability_array = model.predict_proba(input_final)[0]
        
        # Get the probability of the predicted class
        # If prediction is 0 (no default), show probability of no default
        # If prediction is 1 (default), show probability of default
        if prediction == 0:
            probability = probability_array[0]  # probability of no default
        else:
            probability = probability_array[1]  # probability of default
        
        return render_template('result.html', prediction=int(prediction), probability=float(probability))
        
    except Exception as e:
        return render_template('error.html', error_message=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (returns JSON)"""
    try:
        if model is None or scaler is None or feature_names is None:
            return jsonify({
                'success': False,
                'error': 'Model artifacts are unavailable or incompatible on server.',
                'details': MODEL_LOAD_ERROR
            }), 500

        data = request.get_json()
        
        # Create input DataFrame
        input_data = pd.DataFrame([data])

        input_final = preprocess_input(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_final)[0])
        probability = model.predict_proba(input_final)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'probability_default': float(probability[1]),
            'probability_no_default': float(probability[0])
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html')


@app.route('/stats')
def stats():
    """Statistics page with model metrics"""
    return render_template('stats.html')




if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏦 LOAN DEFAULT PREDICTION WEB APPLICATION")
    print("="*60)
    print("Starting Flask server...")
    print("="*60 + "\n")

    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)