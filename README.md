# 💳 Credit Scoring & Loan Decision System

## Project Overview

This is an interactive **Streamlit web application** for **credit risk assessment and prediction**.
It predicts the likelihood of a customer defaulting on a loan using **Machine Learning models** (XGBoost, Logistic Regression) and provides actionable insights for loan approval decisions.

The app provides:

* **Real-time credit risk predictions** using pre-trained ML models
* **Interactive Streamlit dashboard** for easy credit scoring
* **Batch CSV upload** for scoring multiple customers at once
* **Pre-trained ML models** (XGBoost, Logistic Regression) and feature scaler for fast inference
* **Comprehensive logging and exception handling** for robust production deployment
* **Containerized deployment** using Docker

The project demonstrates:

* Python for **data preprocessing, feature engineering, and modeling**
* Pandas & NumPy for **data wrangling and transformation**
* XGBoost & Scikit-learn for **classification and probability predictions**
* Streamlit & Matplotlib for **interactive dashboards and visualizations**
* Custom pipeline architecture for **modular and scalable design**
* Docker for **containerized deployment**

---

## 🚀 Live Demo

Check out the live dashboard here:

[🔗 Open in Streamlit](https://credit-scoring-model-hkbhpf34xqinyaohzw7sxp.streamlit.app/)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- pip or conda

### Installation

**1. Clone the repository:**
```bash
git clone <your-repo-url>
cd credit-scoring-model
```

**2. Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit app:**
```bash
streamlit run creditscoringapp.py
```

The app will open at `http://localhost:8501`

### Docker Deployment

**Build and run with Docker:**
```bash
docker build -t credit-scoring-app .
docker run -p 8501:8501 credit-scoring-app
```


### Streamlit Cloud Deployment

**Deploy to Streamlit Cloud (Recommended for quick deployment):**

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud"
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app" → Select your repository
   - Set:
     - **Main file path:** `creditscoringapp.py`
     - **Python version:** 3.11

3. **Deploy:**
   - Click "Deploy" and wait for the app to build
   - Share your public Streamlit Cloud URL

**Note:** All model artifacts (`model.pkl`, `preprocessor.pkl`) must be in the `artifacts/` directory for the app to work properly on Streamlit Cloud.

---

## Use Case

This app is ideal for:

* **Banks and lenders** assessing loan applications and managing credit risk
* **Data analysts** exploring credit risk patterns and model performance
* **Financial institutions** automating loan approval decisions
* **Credit risk managers** monitoring customer default probability

---

## Features

### ✅ Automated Credit Risk Prediction

* Predict default probability for each customer in real-time
* Generate actionable insights for loan approval decisions
* Score individual or batch customers

### ✅ Interactive Web Dashboard

* User-friendly Streamlit interface
* Sidebar input for customer credit details
* Real-time prediction results and risk classification

### ✅ Batch Predictions

* Upload a CSV file with multiple customers
* Get predictions for all records at once
* Download results as CSV

### ✅ Pre-trained & Optimized Models

* **XGBoost** – Gradient boosting model optimized for credit scoring with high accuracy
* **Logistic Regression** – Interpretable linear model for baseline credit risk classification

### ✅ Feature Engineering & Scaling

* StandardScaler for feature normalization
* Engineered features capturing credit behavior patterns
* Data transformation pipeline for consistent preprocessing

### ✅ Robust Architecture

* Custom exception handling with detailed error messages
* Comprehensive logging system for monitoring and debugging
* Modular pipeline design for easy maintenance and updates

---

## Project Structure

```
credit-scoring-model/
├── creditscoringapp.py              # Streamlit web application
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── setup.py                         # Package setup
├── runtime.txt                      # Python runtime version
├── README.md                        # Project documentation
│
├── artifacts/                       # Pre-trained models and data
│   ├── model.pkl                   # Trained ML model (XGBoost / Logistic Regression)
│   ├── preprocessor.pkl            # Feature scaler (StandardScaler)
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Testing dataset
│   └── data.csv                    # Full dataset
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions
│   ├── components/                # Data pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Load and ingest data
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py       # Model training
│   └── pipeline/                  # Prediction pipelines
│       ├── __init__.py
│       ├── train_pipeline.py     # Training workflow
│       └── predict_pipeline.py   # Prediction workflow
│
├── notebook/                      # Jupyter notebooks
│   ├── EDA CREDIT SCORING.IPYNB        # Exploratory data analysis
│   ├── MODEL TRAINING.ipynb            # Model development
│   └── data/                           # Notebook datasets
│       └── credit_data.csv
│
├── logs/                          # Application logs
└── credit_scoring_model.egg-info/ # Package metadata
```

---

## Dataset

The dataset is based on the **GiveMeSomeCredit** benchmark dataset with the following features:

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit card and personal line utilization ratio |
| `age` | Customer age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times 30–59 days past due |
| `DebtRatio` | Monthly debt payments divided by monthly income |
| `MonthlyIncome` | Customer monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and lines of credit |
| `NumberOfTimes90DaysLate` | Number of times 90+ days past due |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times 60–89 days past due |
| `NumberOfDependents` | Number of dependents in the family |

**Target Variable:** `SeriousDlqin2yrs` — Binary classification (1 = High Risk / 0 = Low Risk)

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Machine Learning | XGBoost, Scikit-learn (Logistic Regression) |
| Preprocessing | StandardScaler, Feature Engineering |
| Web App | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | dill, joblib |
| Notebook Analysis | Jupyter Notebook |
| Deployment | Docker, Streamlit Cloud |
| Logging & Error | Custom Logger, Exception Handling |

---

## Workflow Architecture

```
Raw Credit Data
    ↓
Data Ingestion (components/data_ingestion.py)
    ↓
Data Transformation & Feature Engineering (components/data_transformation.py)
    ↓
Model Training — XGBoost & Logistic Regression (components/model_trainer.py)
    ↓
Pre-trained Model Artifacts (artifacts/)
    ↓
Streamlit App (creditscoringapp.py)
    ↓
Real-time Predictions + Risk Classification
```

### Key Pipeline Steps:

1. **Data Ingestion** – Load credit data from CSV files
2. **Data Transformation** – Feature scaling using StandardScaler
3. **Model Training** – Train XGBoost and Logistic Regression models with cross-validation; best model saved
4. **Prediction** – Score new customers using the best trained model
5. **Result** – Display prediction and risk classification in the Streamlit dashboard

---

## How to Use the App

### Single Customer Prediction

1. Open the Streamlit app at [🔗 Open in Streamlit](https://credit-scoring-model-hkbhpf34xqinyaohzw7sxp.streamlit.app/)
2. Enter customer credit details in the sidebar:
   - Revolving Utilization Of Unsecured Lines
   - Age
   - Debt Ratio
   - Monthly Income
   - Number of Dependents
   - And other credit metrics
3. Click **"Predict Credit Risk"**
4. View the prediction result, default probability, and risk classification

### Batch Prediction

1. Prepare a CSV file with the required columns (see Dataset section above)
2. Upload the CSV using the **"Batch Prediction"** file uploader
3. View predictions for all customers in the table
4. Click **"Download Predictions CSV"** to save results

**Required CSV columns:**
```
RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30-59DaysPastDueNotWorse,
DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse, NumberOfDependents
```

---

## Model Performance

| Model | Description |
|---|---|
| **XGBoost** | Gradient boosted trees; handles imbalanced data well, high predictive accuracy |
| **Logistic Regression** | Fast, interpretable linear baseline; effective for linearly separable patterns |

- **Validation:** K-Fold Cross-Validation
- **Feature Scaling:** StandardScaler
- **Target:** Binary classification — Default (1) / No Default (0)
- **Best model** is automatically selected and saved to `artifacts/model.pkl`

---

## Logging

Logs are stored in the `logs/` directory for monitoring application behavior and debugging.

See `src/logger.py` for logging configuration.

---

## Exception Handling

Custom exception handling is implemented in `src/exception.py` to provide detailed error messages with script name and line number for easier debugging.

---

## Future Enhancements

- [ ] SHAP feature interpretability visualization
- [ ] Model performance metrics dashboard
- [ ] API endpoint deployment (FastAPI)
- [ ] Real-time model retraining pipeline
- [ ] Integration with cloud storage (AWS S3 / Cloudflare R2)
- [ ] Multi-model ensemble predictions

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/your-feature`)
3. Make your changes and test thoroughly
4. Submit a pull request

---

## Contact & Support

For questions or issues, please open a GitHub issue or contact the project maintainer.

---

## Acknowledgments

- Dataset sourced from the **GiveMeSomeCredit** Kaggle competition
- XGBoost and Scikit-learn for powerful ML implementations
- Streamlit for making it easy to build interactive data apps

---

**Last Updated:** May 2026
**Version:** 1.0
