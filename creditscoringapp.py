import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

from src.utils import load_object

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData



# Page Configuration

st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_prediction_pipeline():
    """Cache the prediction pipeline to avoid reloading on every interaction"""
    return PredictPipeline()


# Check if model artifacts exist
def check_artifacts_exist():
    """Check if required model artifacts are available"""
    model_path = 'artifacts/model.pkl'
    preprocessor_path = 'artifacts/preprocessor.pkl'
    return os.path.exists(model_path) and os.path.exists(preprocessor_path)


# Title

st.title("💳 Credit Scoring Prediction System")

st.markdown(
    """
    This application predicts whether a customer is High Risk or Low Risk
    using Machine Learning.
    """
)


# Sidebar

st.sidebar.header("Enter Customer Details")


# User Input

RevolvingUtilizationOfUnsecuredLines = st.sidebar.number_input(
    "Revolving Utilization Of Unsecured Lines",
    min_value=0.0,
    value=0.5
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

NumberOfTime30_59DaysPastDueNotWorse = st.sidebar.number_input(
    "Number Of Time 30-59 Days Past Due Not Worse",
    min_value=0,
    value=0
)

DebtRatio = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.3
)

MonthlyIncome = st.sidebar.number_input(
    "Monthly Income",
    min_value=0.0,
    value=50000.0
)

NumberOfOpenCreditLinesAndLoans = st.sidebar.number_input(
    "Number Of Open Credit Lines And Loans",
    min_value=0,
    value=5
)

NumberOfTimes90DaysLate = st.sidebar.number_input(
    "Number Of Times 90 Days Late",
    min_value=0,
    value=0
)

NumberRealEstateLoansOrLines = st.sidebar.number_input(
    "Number Real Estate Loans Or Lines",
    min_value=0,
    value=1
)

NumberOfTime60_89DaysPastDueNotWorse = st.sidebar.number_input(
    "Number Of Time 60-89 Days Past Due Not Worse",
    min_value=0,
    value=0
)

NumberOfDependents = st.sidebar.number_input(
    "Number Of Dependents",
    min_value=0,
    value=0
)


# Create DataFrame

input_df = pd.DataFrame({

    "RevolvingUtilizationOfUnsecuredLines": [
        RevolvingUtilizationOfUnsecuredLines
    ],

    "age": [
        age
    ],

    "NumberOfTime30-59DaysPastDueNotWorse": [
        NumberOfTime30_59DaysPastDueNotWorse
    ],

    "DebtRatio": [
        DebtRatio
    ],

    "MonthlyIncome": [
        MonthlyIncome
    ],

    "NumberOfOpenCreditLinesAndLoans": [
        NumberOfOpenCreditLinesAndLoans
    ],

    "NumberOfTimes90DaysLate": [
        NumberOfTimes90DaysLate
    ],

    "NumberRealEstateLoansOrLines": [
        NumberRealEstateLoansOrLines
    ],

    "NumberOfTime60-89DaysPastDueNotWorse": [
        NumberOfTime60_89DaysPastDueNotWorse
    ],

    "NumberOfDependents": [
        NumberOfDependents
    ]

})


# Display User Input

st.subheader("Customer Input Data")

st.dataframe(input_df)


# Prediction Button

if st.button("Predict Credit Risk"):

    if not check_artifacts_exist():
        st.error("❌ Error: Model artifacts not found. Please train the model first.")
        st.info("Required files: artifacts/model.pkl and artifacts/preprocessor.pkl")
    else:
        try:

            # Create Custom Data Object
            data = CustomData(

                RevolvingUtilizationOfUnsecuredLines=RevolvingUtilizationOfUnsecuredLines,

                age=age,

                NumberOfTime30_59DaysPastDueNotWorse=NumberOfTime30_59DaysPastDueNotWorse,

                DebtRatio=DebtRatio,

                MonthlyIncome=MonthlyIncome,

                NumberOfOpenCreditLinesAndLoans=NumberOfOpenCreditLinesAndLoans,

                NumberOfTimes90DaysLate=NumberOfTimes90DaysLate,

                NumberRealEstateLoansOrLines=NumberRealEstateLoansOrLines,

                NumberOfTime60_89DaysPastDueNotWorse=NumberOfTime60_89DaysPastDueNotWorse,

                NumberOfDependents=NumberOfDependents

            )


            # Convert into DataFrame
            pred_df = data.get_data_as_data_frame()


            # Prediction Pipeline (cached)
            predict_pipeline = load_prediction_pipeline()

            prediction = predict_pipeline.predict(pred_df)

            #Probability Prediction
            probability = predict_pipeline.predict_proba(pred_df)

            risk_probability = probability[0][1] * 100


            # Prediction Result
            st.subheader("Prediction Result")

            if prediction[0] == 1:

                st.error("⚠️ High Risk Customer")

                st.progress(90)

            else:

                st.success("✅ Low Risk Customer")

                st.progress(20)
            
            # probability Result
            st.metric(
                label="Default Probability",
                value=f"{risk_probability:.2f}%"
            )

            if risk_probability < 30:

                st.success("🟢 Low Default Risk")

            elif risk_probability < 70:

                st.warning("🟡 Medium Default Risk")

            else:

                st.error("🔴 High Default Risk")


        
            # SHAP Explainability
            st.subheader("SHAP Explainability")

            model = load_object("artifacts/model.pkl")
            preprocessor = load_object("artifacts/preprocessor.pkl")

            input_scaled = preprocessor.transform(input_df)

            explainer = shap.Explainer(model, input_scaled)

            shap_values = explainer(input_scaled)

            fig, ax = plt.subplots(figsize=(10, 5))

            shap.plots.bar(
                shap_values[0],
                show=False
            )

            st.pyplot(fig)


        except Exception as e:

            st.error(f"Error Occurred: {str(e)}")
            st.info("Please check that all input values are valid and try again.")


# Batch Prediction Section

st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    if not check_artifacts_exist():
        st.error("❌ Error: Model artifacts not found. Please train the model first.")
        st.info("Required files: artifacts/model.pkl and artifacts/preprocessor.pkl")
    else:
        try:

            batch_df = pd.read_csv(uploaded_file)

            st.write("Uploaded Dataset")

            st.dataframe(batch_df.head())


            # Prediction Pipeline (cached)
            predict_pipeline = load_prediction_pipeline()

            batch_predictions = predict_pipeline.predict(batch_df)


            # Add Prediction Column
            batch_df["Prediction"] = batch_predictions


            # Convert 0/1 into Labels
            batch_df["Prediction"] = batch_df["Prediction"].map({
                0: "Low Risk",
                1: "High Risk"
            })


            st.subheader("Batch Prediction Result")

            st.dataframe(batch_df)


            # Download Button
            csv = batch_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="credit_predictions.csv",
                mime="text/csv"
            )


        except Exception as e:

            st.error(f"Batch Prediction Error: {str(e)}")
            st.info("Please ensure the CSV file has the correct format with all required columns.")
