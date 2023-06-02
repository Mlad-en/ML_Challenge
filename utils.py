import warnings
from typing import Any

from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate, GridSearchCV, train_test_split, RandomizedSearchCV
)

from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score, make_scorer

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

    def full_grid_search(self, model, parameters: dict[str, list[Any]]):
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
            scoring=self.scoring,
        )

        return self
    
    def random_grid_search(self, model, parameters: dict[str, list[Any]]):
        """
        Perform grid search for hyperparameter tuning.

        :param model: Model to be tuned.
        :param parameters: Hyperparameter grid.
        :return: TuneHyperParams utility object.
        """
        self.gs = RandomizedSearchCV(
            model,
            parameters,
            n_jobs=self.jobs,
            cv=self.cross_validation,
            scoring=self.scoring,
            random_state=ModelConstants.RANDOM_STATE,
            verbose=2
        )

        return self

    def fit_model(self, predictors, target):
        """
        Fit the model using the provided predictors and target.

        :param predictors: Predictors (input features) for model training.
        :param target: Target variable for model training.
        :return: TuneHyperParams utility object.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gs.fit(predictors, target)
            return self

    def get_best_model(self):
        """
        Print the best parameters found during grid search and return the grid search object.
        :return: Grid search object.
        """
        print(self.gs.best_params_)
        print(f"Best parameter (CV score: {self.gs.best_score_:.3f}):")
        return self.gs.best_estimator_


def get_cross_validation_results(model, predictors, outcome):
    """
    Perform cross-validation and calculate scoring metrics for the model.

    :param model: Model to be evaluated.
    :param predictors: Predictors (input features) for cross-validation.
    :param outcome: Outcome (target variable) for cross-validation.
    :return: Dictionary of scoring metrics.
    """
    matthews_score = make_scorer(matthews_corrcoef)

    scoring_params = {
        "F1 Score": "f1",
        "Accuracy": "accuracy",
        "Balanced Accuracy": "balanced_accuracy",
        "Matthew's Correlation Coefficient": matthews_score
    }

    scores = cross_validate(
        model,
        predictors,
        outcome,
        cv=ModelConstants.CROSS_VALIDATIONS,
        scoring=scoring_params
    )

    test = []
    score = []

    for score_type, score_value in scores.items():
        test.append(score_type.replace("test_", f"{ModelConstants.CROSS_VALIDATIONS}-fold CV ") + " mean score")
        score.append(score_value.mean())

    return pd.DataFrame(
        {
            "Metric for Training Set": test,
            "Score": score
        }
    )


def get_final_model_performance(
        model,
        training_data: DataSplit,
        testing_data: DataSplit
):

    model.fit(training_data.predictors, training_data.outcome)
    predictions = model.predict(testing_data.predictors)

    scoring_params = {
        "F1 Score": f1_score,
        "Accuracy": accuracy_score,
        "Balanced Accuracy": balanced_accuracy_score,
        "Matthew's Correlation Coefficient": matthews_corrcoef
    }
    test = []
    score = []

    for score_type, func in scoring_params.items():
        test.append(score_type)
        score.append(func(predictions, testing_data.outcome))

    return pd.DataFrame(
        {
            "Metric for Testing Set": test,
            "Score": score
        }
    )