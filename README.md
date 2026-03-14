# 🏦 Loan Default Prediction - ML Project

## Darshan University - AI/ML Subject

A comprehensive Machine Learning project for predicting loan defaults using various ML algorithms.

---

## 📋 Project Tasks Completed

| Task | Title | Status |
|------|-------|--------|
| 1 | Problem Definition and Dataset Exploration | ✅ |
| 2 | Data Cleaning and Pre-processing | ✅ |
| 3 | Model Creation (From Scratch + Library) | ✅ |
| 4 | Model Evaluation | ✅ |
| 5 | Advanced Model Training | ✅ |
| 6 | Visualization of Metrics | ✅ |
| 7 | Flask Project Setup | ✅ |
| 8 | Create Front End | ✅ |
| 9 | Create Backend | ✅ |
| 10 | Deployment | ✅ (Ready) |

---

## 🚀 How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Jupyter Notebook
Open `Loan_Default_ML_Project.ipynb` and run all cells to:
- Explore and preprocess the data
- Train models
- Generate model files (model.pkl, scaler.pkl, etc.)

### Step 3: Run the Flask Application
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

## 📁 Project Structure

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

## 📊 Dataset Features

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

## 🤖 Models Implemented

1. **Logistic Regression** (From Scratch + Sklearn)
2. **Decision Tree Classifier**
3. **Random Forest Classifier** ⭐ Best Model
4. **Gradient Boosting Classifier**
5. **K-Nearest Neighbors**

---

## 🌐 Deployment (Task 10)

### Option 1: Heroku (Free)
```bash
# Install Heroku CLI, then:
heroku login
heroku create your-app-name
git push heroku main
```

### Option 2: Render.com (Free)
1. Connect your GitHub repository
2. Select Python environment
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`

### Option 3: PythonAnywhere (Free)
1. Upload project files
2. Create a web app with Flask
3. Configure WSGI file

---

## 👨‍🎓 Author

**Darshan University Student**  
AI/ML Project - Loan Default Prediction

---

## 📝 License

This project is for educational purposes only.
