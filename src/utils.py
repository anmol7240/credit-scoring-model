import os
import sys

import dill
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_models(

    x_train,
    y_train,

    x_test,
    y_test,

    models,
    param

):

    try:

        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]

            para = param[list(models.keys())[i]]

            # Hyperparameter Tuning
            gs = GridSearchCV(

                estimator=model,

                param_grid=para,

                cv=3,

                scoring='roc_auc',

                n_jobs=-1,

                verbose=1

            )

            # Train GridSearch
            gs.fit(
                x_train,
                y_train
            )

            # Best Parameters
            model.set_params(
                **gs.best_params_
            )

            # Train Model
            model.fit(
                x_train,
                y_train
            )

            # Predictions
            y_train_pred = model.predict(
                x_train
            )

            y_test_pred = model.predict(
                x_test
            )

            # ROC-AUC Score
            train_model_score = roc_auc_score(
                y_train,
                y_train_pred
            )

            test_model_score = roc_auc_score(
                y_test,
                y_test_pred
            )

            # Store model score
            report[
                list(models.keys())[i]
            ] = test_model_score

        return report

    except Exception as e:

        raise CustomException(e, sys)


def load_object(file_path):

    try:

        with open(file_path, 'rb') as file_obj:

            return dill.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)