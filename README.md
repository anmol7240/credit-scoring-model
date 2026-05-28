"# 💳 Credit Scoring & Loan Decision System

## Project Overview

This is an interactive **Streamlit web application** for **credit risk assessment and prediction**.
It predicts the likelihood of a customer defaulting on a loan using **Machine Learning models** (CatBoost, XGBoost, Logistic Regression) and provides actionable insights for loan approval decisions.

The app provides:

* **Real-time credit risk predictions** using pre-trained ML models
* **Interactive Streamlit dashboard** for easy credit scoring
* **Batch CSV upload** for scoring multiple customers at once
* **Pre-trained ML models** (CatBoost, XGBoost) and feature scaler for fast inference
* **Comprehensive logging and exception handling** for robust production deployment
* **Containerized deployment** using Docker

The project demonstrates:

* Python for **data preprocessing, feature engineering, and modeling**
* Pandas & NumPy for **data wrangling and transformation**
* CatBoost, XGBoost & Scikit-learn for **classification and probability predictions**
* Streamlit & Matplotlib for **interactive dashboards and visualizations**
* Custom pipeline architecture for **modular and scalable design**
* Docker for **containerized deployment**

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

Then access at: **http://localhost:8501**

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

**Note:** All model artifacts (model.pkl, preprocessor.pkl) must be in the `artifacts/` directory for the app to work properly on Streamlit Cloud.

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

* Upload custom CSVs to score multiple customers at once
* Download predictions for further analysis

### ✅ Pre-trained & Optimized Models

* **CatBoost** - Gradient boosting model optimized for credit scoring
* **XGBoost** - Alternative boosting model for comparison
* Models trained on historical credit data with cross-validation

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
├── setup.py                        # Package setup
├── README.md                       # Project documentation
│
├── artifacts/                      # Pre-trained models and data
│   ├── model.pkl                  # Trained ML model
│   ├── preprocessor.pkl           # Feature scaler
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Testing dataset
│   └── data.csv                   # Full dataset
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── exception.py              # Custom exception handling
│   ├── logger.py                 # Logging configuration
│   ├── utils.py                  # Utility functions
│   ├── components/               # Data pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Load and ingest data
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py      # Model training
│   └── pipeline/                # Prediction pipelines
│       ├── __init__.py
│       ├── train_pipeline.py    # Training workflow
│       └── predict_pipeline.py  # Prediction workflow
│
├── notebook/                     # Jupyter notebooks
│   ├── EDA CREDIT SCORING.IPYNB       # Exploratory data analysis
│   ├── MODEL TRAINING.ipynb           # Model development
│   └── data/                          # Notebook datasets
│
├── logs/                        # Application logs
├── catboost_info/               # CatBoost training info
└── credit_scoring_model.egg-info/  # Package metadata
```

---

## Dataset

The dataset consists of historical credit information with the following features:

* `RevolvingUtilizationOfUnsecuredLines` - Credit utilization ratio
* `Age` - Customer age
* `NumberOfTime30_59DaysPastDueNotWorse` - Late payments (30-59 days)
* `DebtRatio` - Debt-to-income ratio
* `MonthlyIncome` - Customer monthly income
* `NumberOfOpenCreditLinesAndLoans` - Open credit accounts
* `NumberOfTimes90DaysLate` - Late payments (90+ days)
* `NumberRealEstateLoansOrLines` - Real estate credit lines
* `NumberOfTime60_89DaysPastDueNotWorse` - Late payments (60-89 days)
* `NumberOfDependents` - Number of dependents

**Target Variable:** Credit default (High Risk / Low Risk)

---

## Tech Stack

| Category          | Tools                                       |
| ----------------- | ------------------------------------------- |
| Programming       | Python 3.10                                 |
| Data Handling     | Pandas, NumPy                               |
| Machine Learning  | CatBoost, XGBoost, Scikit-learn             |
| Preprocessing     | StandardScaler, Feature Engineering         |
| Web App           | Streamlit                                   |
| Visualization     | Matplotlib, Seaborn                         |
| Model Persistence | dill, joblib                                |
| Notebook Analysis | Jupyter Notebook                            |
| Deployment        | Docker, Streamlit Cloud                     |
| Logging & Error   | Custom Logger, Exception Handling           |

---

## Workflow Architecture

```
Raw Credit Data 
    ↓
Data Ingestion (components/data_ingestion.py)
    ↓
Data Transformation & Feature Engineering (components/data_transformation.py)
    ↓
Model Training (components/model_trainer.py)
    ↓
Pre-trained Model Artifacts (artifacts/)
    ↓
Streamlit App (creditscoringapp.py)
    ↓
Real-time Predictions + Risk Classification
```

### Key Pipeline Steps:

1. **Data Ingestion** – Load credit data from CSV files
2. **Data Transformation** – Feature scaling and engineering using StandardScaler
3. **Model Training** – Train CatBoost/XGBoost models with cross-validation
4. **Prediction** – Score new customers using the trained models
5. **Result** – Display prediction and risk classification in Streamlit dashboard

---

## How to Use the App

### Single Customer Prediction

1. Open the Streamlit app
2. Enter customer credit details in the sidebar:
   - Revolving Utilization Of Unsecured Lines
   - Age
   - Debt Ratio
   - Monthly Income
   - Number of Dependents
   - And other credit metrics
3. Click **"Predict Credit Risk"** button
4. View the prediction result and risk classification

### Batch Predictions (Future Enhancement)

- Upload a CSV file with multiple customers
- Get predictions for all records at once
- Download results

---

## Model Performance

- **Models Used:** CatBoost, XGBoost
- **Validation:** K-Fold Cross-Validation
- **Feature Scaling:** StandardScaler (StandardScaler for feature normalization)
- **Target:** Binary classification (Default/No Default)

---

## Environment Variables

No additional environment variables required. The app uses default configurations.

---

## Logging

Logs are stored in the `logs/` directory for monitoring application behavior and debugging issues.

Check `src/logger.py` for logging configuration.

---

## Exception Handling

Custom exception handling is implemented in `src/exception.py` to provide detailed error messages and stack traces for debugging.

---

## Future Enhancements

- [ ] SHAP feature interpretability visualization
- [ ] Batch CSV upload functionality
- [ ] Model performance metrics dashboard
- [ ] API endpoint deployment
- [ ] Real-time model retraining pipeline
- [ ] Multi-model ensemble predictions
- [ ] Integration with Cloudflare R2 for cloud storage

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

---

## Contact & Support

For questions or issues, please open a GitHub issue or contact the project maintainer.

---

## Acknowledgments

- Data preprocessing inspired by credit risk modeling best practices
- CatBoost and XGBoost for powerful gradient boosting implementations
- Streamlit for making it easy to build interactive data apps

---

**Last Updated:** May 2026  
**Version:** 1.0" 
