from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split

from constants import Columns, ModelConstants

DATA_TYPES = {
    Columns.TRANSACTION: np.int8,
    Columns.CUSTOMER_TYPE: "category",
    Columns.SYSTEM_F1: np.int8,
    Columns.SYSTEM_F2: np.int8,
    Columns.SYSTEM_F3: np.int8,
    Columns.SYSTEM_F4: np.int8,
    Columns.SYSTEM_F5: np.int8,
    Columns.ACCOUNT_PAGE: np.int8,
    Columns.ACCOUNT_PAGE_TIME: np.float64,
    Columns.INFO_PAGE: np.int8,
    Columns.INFO_PAGE_TIME: np.float64,
    Columns.PRODUCT_PAGE: np.int16,
    Columns.PRODUCT_PAGE_TIME: np.float64,
    Columns.MONTH: np.int8,
    Columns.WEEKDAY: "bool",
    Columns.SPECIFIC_HOLIDAY: "category",
    Columns.GOOGLE_ANALYTICS_BR: np.float64,
    Columns.GOOGLE_ANALYTICS_ER: np.float64,
    Columns.GOOGLE_ANALYTICS_PV: np.float64,
    Columns.AD_CAMPAIGN_1: np.int8,
    Columns.AD_CAMPAIGN_2: np.int8,
    Columns.AD_CAMPAIGN_3: np.int8,
}


class DataSplit:
    """
    Represents a split of data into predictors and outcomes.
    """

    def __init__(self, x, y):
        self.predictors = x
        self.outcome = y


class Data:
    """
    Represents training and testing data splits.
    """

    def __init__(self, x_train, x_test, y_train, y_test):
        self.TRAINING = DataSplit(x_train, y_train)
        self.TESTING = DataSplit(x_test, y_test)


class TransactionDataset:
    """
    Represents the ML Challenge Transaction dataset.
    """

    _CSV_FILE = "https://raw.githubusercontent.com/Mlad-en/ML_Challenge/main/Newdata-2.csv"

    def __init__(self):
        self.data = pd.read_csv(self._CSV_FILE, dtype=DATA_TYPES)

    @property
    def _target(self):
        return self.data[Columns.TRANSACTION]

    @property
    def _predictors(self):
        include_cols = [col for col in self.data.columns if not col == Columns.TRANSACTION]
        return self.data.loc[:, include_cols]

    def get_training_test_split(self):
        """
        Split the dataset into training and testing sets.

        :return: Data object containing the training and testing splits.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            self._predictors,
            self._target,
            stratify=self._target,
            test_size=.2,
            random_state=ModelConstants.RANDOM_STATE
        )

        return Data(x_train, x_test, y_train, y_test)


class TuneHyperParams:
    """
    Utility class for hyperparameter tuning.
    """

    def __init__(self, cross_validation: int = 5, jobs: int = 5, scoring: str = ModelConstants.F1_SCORE):
        """
        Initialize the TuneHyperParams object.

        :param cross_validation: Number of cross-validation folds.
        :param jobs: Number of jobs to run in parallel during grid search.
        :param scoring: Scoring metric used for model evaluation.
        """
        self.gs = None
        self.cross_validation = cross_validation
        self.jobs = jobs
        self.scoring = scoring

    def grid_search(self, model, parameters: dict[str, list[Any]]):
        """
        Perform grid search for hyperparameter tuning.

        :param model: Model to be tuned.
        :param parameters: Hyperparameter grid.
        :return: TuneHyperParams utility object.
        """
        self.gs = GridSearchCV(
            model,
            parameters,
            n_jobs=self.jobs,
            cv=self.cross_validation,
            scoring=ModelConstants.F1_SCORE,
        )

        return self

    def fit_model(self, predictors, target):
        """
        Fit the model using the provided predictors and target.

        :param predictors: Predictors (input features) for model training.
        :param target: Target variable for model training.
        :return: TuneHyperParams utility object.
        """
        self.gs.fit(predictors, target)
        return self

    def get_best_scores_and_params(self):
        """
        Print the best parameters found during grid search and return the grid search object.
        :return: Grid search object.
        """
        print(self.gs.best_params_)
        print(f"Best parameter (CV score{self.gs.best_score_}:.3f):")
        return self.gs


def get_cross_validation_results(model, predictors, outcome):
    """
    Perform cross-validation and calculate scoring metrics for the model.

    :param model: Model to be evaluated.
    :param predictors: Predictors (input features) for cross-validation.
    :param outcome: Outcome (target variable) for cross-validation.
    :return: Dictionary of scoring metrics.
    """
    scores = cross_validate(
        model,
        predictors,
        outcome,
        cv=10,
        scoring=[
            ModelConstants.ACCURACY_SCORE,
            ModelConstants.BALANCED_ACCURACY_SCORE,
            ModelConstants.F1_SCORE
        ]
    )

    for score_type, score in scores.items():
        print(f"{score_type}: {score.mean()}")

    return scores
