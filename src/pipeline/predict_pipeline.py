import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = 'artifacts/model.pkl'

            preprocessor_path = 'artifacts/preprocessor.pkl'

            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

            model = load_object(
                file_path=model_path
            )

            preprocessor = load_object(
                file_path=preprocessor_path
            )

            # Add missing "Unnamed: 0" column if it doesn't exist
            # (this column was present during training, so preprocessor expects it)
            if "Unnamed: 0" not in features.columns:
                features.insert(0, "Unnamed: 0", 0)

            # Transform input data
            data_scaled = preprocessor.transform(
                features
            )

            # Prediction
            preds = model.predict(
                data_scaled
            )

            return preds

        except Exception as e:

            raise CustomException(e, sys)


class CustomData:

    def __init__(

        self,

        RevolvingUtilizationOfUnsecuredLines: float,
        age: int,
        NumberOfTime30_59DaysPastDueNotWorse: int,
        DebtRatio: float,
        MonthlyIncome: float,
        NumberOfOpenCreditLinesAndLoans: int,
        NumberOfTimes90DaysLate: int,
        NumberRealEstateLoansOrLines: int,
        NumberOfTime60_89DaysPastDueNotWorse: int,
        NumberOfDependents: int

    ):

        self.RevolvingUtilizationOfUnsecuredLines = RevolvingUtilizationOfUnsecuredLines

        self.age = age

        self.NumberOfTime30_59DaysPastDueNotWorse = NumberOfTime30_59DaysPastDueNotWorse

        self.DebtRatio = DebtRatio

        self.MonthlyIncome = MonthlyIncome

        self.NumberOfOpenCreditLinesAndLoans = NumberOfOpenCreditLinesAndLoans

        self.NumberOfTimes90DaysLate = NumberOfTimes90DaysLate

        self.NumberRealEstateLoansOrLines = NumberRealEstateLoansOrLines

        self.NumberOfTime60_89DaysPastDueNotWorse = NumberOfTime60_89DaysPastDueNotWorse

        self.NumberOfDependents = NumberOfDependents

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "RevolvingUtilizationOfUnsecuredLines": [
                    self.RevolvingUtilizationOfUnsecuredLines
                ],

                "age": [
                    self.age
                ],

                "NumberOfTime30-59DaysPastDueNotWorse": [
                    self.NumberOfTime30_59DaysPastDueNotWorse
                ],

                "DebtRatio": [
                    self.DebtRatio
                ],

                "MonthlyIncome": [
                    self.MonthlyIncome
                ],

                "NumberOfOpenCreditLinesAndLoans": [
                    self.NumberOfOpenCreditLinesAndLoans
                ],

                "NumberOfTimes90DaysLate": [
                    self.NumberOfTimes90DaysLate
                ],

                "NumberRealEstateLoansOrLines": [
                    self.NumberRealEstateLoansOrLines
                ],

                "NumberOfTime60-89DaysPastDueNotWorse": [
                    self.NumberOfTime60_89DaysPastDueNotWorse
                ],

                "NumberOfDependents": [
                    self.NumberOfDependents
                ]

            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:

            raise CustomException(e, sys)