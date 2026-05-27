import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# CatBoost is optional - may not install on some cloud platforms
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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

                "K-Neighbors Classifier": KNeighborsClassifier(),

                "Decision Tree": DecisionTreeClassifier(),

                "Random Forest Classifier": RandomForestClassifier(),

                "Gradient Boosting Classifier": GradientBoostingClassifier(),

                "XGBClassifier": XGBClassifier(),

                "AdaBoost Classifier": AdaBoostClassifier()

            }
            
            # Add CatBoost only if available
            if CATBOOST_AVAILABLE:
                models["CatBoosting Classifier"] = CatBoostClassifier(
                    verbose=False
                )

            params = {

                "Logistic Regression": {

                    "C": [0.01, 0.1, 1, 10]

                },

                "K-Neighbors Classifier": {

                    "n_neighbors": [3, 5, 7, 9]

                },

                "Decision Tree": {

                    "criterion": ["gini", "entropy"],

                    "max_depth": [5, 10, 15, None]

                },

                "Random Forest Classifier": {

                    "n_estimators": [50, 100, 200],

                    "max_depth": [5, 10, None]

                },

                "Gradient Boosting Classifier": {

                    "learning_rate": [0.01, 0.1, 0.5],

                    "n_estimators": [50, 100, 200]

                },

                "XGBClassifier": {

                    "learning_rate": [0.01, 0.1, 0.5],

                    "n_estimators": [50, 100, 200]

                },

                "CatBoosting Classifier": {

                    "learning_rate": [0.01, 0.1],

                    "depth": [4, 6, 8]

                },

                "AdaBoost Classifier": {

                    "learning_rate": [0.01, 0.1, 1],

                    "n_estimators": [50, 100, 200]

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

            # Train best model
            best_model.fit(
                X_train_smote,
                y_train_smote
            )

            # Save model
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