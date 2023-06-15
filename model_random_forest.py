import numpy as np
import pandas as pd
from constants import Columns, ModelConstants, ModelFileNames, Locations
from utils import TransactionDataset, TuneHyperParams, FinalModelPerformance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from joblib import dump


if __name__ == "__main__":
    # Getting the dataset from GitHub and splitting the data

    data = TransactionDataset().get_training_test_split()

    onehot_columns = [Columns.CUSTOMER_TYPE]
    ordinal_columns = [Columns.SPECIFIC_HOLIDAY]

    # Pre-Define the columns transformation that we want to make

    ordinal_and_onehot_transformation = make_column_transformer(
        (OneHotEncoder(), onehot_columns),
        (OrdinalEncoder(), ordinal_columns),
        remainder="passthrough",
    )

    # Pre-Defining the model that we want to use and the over/under-sampling methods

    random_forest = RandomForestClassifier(random_state=ModelConstants.RANDOM_STATE)
    tomek_links = TomekLinks(sampling_strategy="majority")

    rf_pipeline = Pipeline(
        [
            ("column_transforms", ordinal_and_onehot_transformation),
            ("Tomek_Links_UnderSampling", tomek_links),
            ("random_forest", random_forest),
        ]
    )

    # Choosing hyperparameters and justification
    random_params_rf = {
        "random_forest__bootstrap": [True, False],
        # Prevent over-fitting and reduce variance if set to True, check if there is a difference
        "random_forest__criterion": [
            "gini",
            "entropy",
            "log_loss",
        ],  # Testing different split functions
        "random_forest__max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # Testing optimal depth and prevent over-fitting when set to None
        "random_forest__max_features": ["log2", "sqrt"],
        # Number of features to consider when looking for the best split
        "random_forest__min_samples_leaf": [1, 2, 4, 6, 8, 10],
        # Minimum number of samples required to be at a leaf node
        "random_forest__min_samples_split": [2, 5, 10, 15, 20, 25],
        # Minimum number of samples required to split an internal node
        "random_forest__n_estimators": [
            100,
            200,
            400,
            600,
            800,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
        ],
        # Number of trees in the forest, a lot of trees can slow down the training process
        # Takes into account the imbalance in the decision class
        "random_forest__class_weight": [
            "balanced",
            {0: 3, 1: 7},
            {0: 2, 1: 8},
            {0: 1, 1: 9},
            None,
        ],
    }

    # Running a Random Search on the pipeline with the above selected parameters
    # and fitting/evaluating the performance on the training data

    best_model_rf = (
        TuneHyperParams()
        .random_grid_search(rf_pipeline, random_params_rf)
        .fit_model(data.TRAINING.predictors, data.TRAINING.outcome)
        .get_best_model()
    )

    # Cross-validating the best model on the training data to get a better overall look
    # on the performance using different performance metrics

    final_model = FinalModelPerformance(model=best_model_rf, data=data)

    final_model.get_cross_validation_results()
    final_model.get_final_model_performance()

    dump(final_model.model, Locations.models_exports / ModelFileNames.random_forest)
