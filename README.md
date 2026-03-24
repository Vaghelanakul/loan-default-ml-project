# Loan Default Prediction Project

Semester project for AI/ML, focused on predicting whether a loan may default based on applicant and loan details.

---

## What This Project Covers

| Step | Work Done | Status |
|------|-----------|--------|
| 1 | Problem understanding and dataset exploration | Done |
| 2 | Data cleaning and preprocessing | Done |
| 3 | Model building (multiple algorithms) | Done |
| 4 | Model evaluation and comparison | Done |
| 5 | Model finalization (best model selection) | Done |
| 6 | Flask app integration (frontend + backend) | Done |
| 7 | Deployment setup files | Done |

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Jupyter Notebook (first time setup)
Open `Loan_Default_ML_Project.ipynb` and run all cells to:
- Explore and preprocess the data
- Train models
- Generate model files (`model.pkl`, `scaler.pkl`, etc.)

### Step 3: Run the Flask Application
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

## Project Structure

```
loan_ml_project/
│
├── Loan_default.csv              # Dataset
├── Loan_Default_ML_Project.ipynb # Main notebook (Tasks 1-6)
├── app.py                        # Flask application (Tasks 7-9)
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku deployment
├── runtime.txt                   # Python version
├── README.md                     # This file
│
├── templates/                    # HTML Templates
│   ├── index.html               # Home page with form
│   ├── result.html              # Prediction result page
│   ├── error.html               # Error page
│   └── about.html               # About page
│
└── (Generated after running notebook)
    ├── model.pkl                # Trained model
    ├── scaler.pkl               # Feature scaler
    ├── label_encoders.pkl       # Categorical encoders
    └── feature_names.pkl        # Feature names
```

---

## Dataset Features

| Feature | Description |
|---------|-------------|
| Age | Applicant's age |
| Income | Annual income |
| LoanAmount | Requested loan amount |
| CreditScore | Credit score (300-850) |
| MonthsEmployed | Employment duration |
| NumCreditLines | Number of credit lines |
| InterestRate | Loan interest rate |
| LoanTerm | Loan duration (months) |
| DTIRatio | Debt-to-income ratio |
| Education | Education level |
| EmploymentType | Type of employment |
| MaritalStatus | Marital status |
| HasMortgage | Has existing mortgage |
| HasDependents | Has dependents |
| LoanPurpose | Purpose of loan |
| HasCoSigner | Has co-signer |
| **Default** | **Target (0/1)** |

---

## Models Tried

1. **Logistic Regression** (from scratch + scikit-learn)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (best performing)
4. **Gradient Boosting Classifier**
5. **K-Nearest Neighbors**

---

## Deployment Notes

### Option 1: Heroku
```bash
# Install Heroku CLI, then:
heroku login
heroku create your-app-name
git push heroku main
```

### Option 2: Render.com
1. Connect your GitHub repository
2. Select Python environment
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. If your Render service has a Start Command set in dashboard settings, it overrides Procfile. Keep the dashboard command the same as step 4.

### Option 3: PythonAnywhere
1. Upload project files
2. Create a web app with Flask
3. Configure WSGI file

---

## Viva / Demo Flow (Quick)

1. Start from the problem statement: reduce default risk by using historical patterns.
2. Explain preprocessing: handling categories, scaling numeric fields, feature alignment.
3. Show why Random Forest was selected (best score among tested models).
4. Run 2-3 sample predictions from the web form.
5. Show the metrics page and discuss limitations.

## Limitations

- This model should support decision-making, not replace human review.
- Performance depends on how similar new data is to the training data.
- Threshold and false-negative cost can be tuned based on business policy.

## Author

Semester 6 AI/ML Project

---

## License

This project is for educational purposes only.
