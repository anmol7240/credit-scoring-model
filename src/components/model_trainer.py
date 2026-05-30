import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],
                train_array[:, -1],

                test_array[:, :-1],
                test_array[:, -1]

            )

            logging.info(
                "Applying SMOTE for handling imbalanced data"
            )

            smote = SMOTE(random_state=42)

            X_train_smote, y_train_smote = smote.fit_resample(
                X_train,
                y_train
            )

            models = {

                "Logistic Regression": LogisticRegression(),

                "XGBClassifier": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                )

            }

            params = {

                "Logistic Regression": {

                    "C": [0.01, 0.1, 1, 10]

                },

                "XGBClassifier": {

                    "learning_rate": [0.01, 0.1],

                    "n_estimators": [50, 100],

                    "max_depth": [3, 5]

                }

            }

            model_report: dict = evaluate_models(

                x_train=X_train_smote,
                y_train=y_train_smote,

                x_test=X_test,
                y_test=y_test,

                models=models,
                param=params

            )

            # Best Model Score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Best Model Name
            best_model_name = list(model_report.keys())[

                list(model_report.values()).index(
                    best_model_score
                )

            ]

            # Best Model
            best_model = models[best_model_name]

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found",
                    sys
                )

            logging.info(
                f"Best model found: {best_model_name}"
            )

            # Train Best Model
            best_model.fit(
                X_train_smote,
                y_train_smote
            )

            # Save Model
            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model

            )

            logging.info(
                "Best model saved successfully"
            )

            # Prediction
            predicted = best_model.predict(X_test)

            # ROC-AUC Score
            roc_score = roc_auc_score(
                y_test,
                predicted
            )

            return roc_score

        except Exception as e:

            raise CustomException(e, sys)