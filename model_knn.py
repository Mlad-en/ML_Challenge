from utils import TransactionDataset, TuneHyperParams, FinalModelPerformance

from constants import Columns, ModelConstants, ResamplingStrategy, ModelFileNames, Locations

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    Normalizer,
    PowerTransformer,
    RobustScaler,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.compose import make_column_transformer, make_column_selector

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss

from joblib import dump


if __name__ == '__main__':

    data = TransactionDataset().get_training_test_split()

    pages_to_drop = [
                Columns.INFO_PAGE_TIME,
                Columns.PRODUCT_PAGE_TIME,
                Columns.GOOGLE_ANALYTICS_ER,
            ]

    log_transform = FunctionTransformer(
        func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )

    order_mapping = [
        ["0", "0.2", "0.4", "0.6", "0.8", "1"],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]

    norm_and_one_hot_transforms = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (
            OrdinalEncoder(categories=order_mapping),
            [Columns.SPECIFIC_HOLIDAY, Columns.MONTH],
        ),
        ("drop", pages_to_drop),
        (Normalizer(), make_column_selector(dtype_include="number")),
    )

    log_and_ordinal_transforms = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (
            OrdinalEncoder(categories=order_mapping),
            [Columns.SPECIFIC_HOLIDAY, Columns.MONTH],
        ),
        ("drop", pages_to_drop),
        (log_transform, make_column_selector(dtype_include="number")),
    )

    power_transforms = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (
            OrdinalEncoder(categories=order_mapping),
            [Columns.SPECIFIC_HOLIDAY, Columns.MONTH],
        ),
        ("drop", pages_to_drop),
        (PowerTransformer(), make_column_selector(dtype_include="number")),
    )

    robust_transformation = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (
            OrdinalEncoder(categories=order_mapping),
            [Columns.SPECIFIC_HOLIDAY, Columns.MONTH],
        ),
        ("drop", pages_to_drop),
        (RobustScaler(), make_column_selector(dtype_include="number")),
    )

    knn_classifier = KNeighborsClassifier(n_jobs=ModelConstants.N_JOBS)

    smote_sampling = SMOTE(random_state=ModelConstants.RANDOM_STATE)
    adasyn_sampling = ADASYN(random_state=ModelConstants.RANDOM_STATE)

    near_miss_sampling = NearMiss()
    tomek_link_sampling = TomekLinks()

    select_best_features = SelectKBest(score_func=mutual_info_classif)

    column_transforms_only_model = Pipeline(
        [
            ("column_transformation", log_and_ordinal_transforms),
            ("over_sampling", smote_sampling),
            ("under_sampling", near_miss_sampling),
            ("feature_selection", select_best_features),
            ("knn", knn_classifier),
        ]
    )

    column_transformation_tune = {
        "column_transformation": [
            log_and_ordinal_transforms,
            norm_and_one_hot_transforms,
            power_transforms,
            robust_transformation,
        ]
    }

    over_sampling_tuning = {
        "over_sampling": [
            smote_sampling,
            adasyn_sampling,
        ]
    }

    over_sampling__sampling_strategy_tuning = {
        "over_sampling__sampling_strategy": [
            ResamplingStrategy.MINORITY_ONLY,
            ResamplingStrategy.ALL,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
        ],
    }

    under_sampling_tuning = {"under_sampling": [near_miss_sampling, tomek_link_sampling, "passthrough"],}
    knn__n_neighbors_tuning = {"knn__n_neighbors": [5, 7, 9, 11, 13, 15, 17, 19, 21, 25]}
    knn__weights = {"knn__weights": ["uniform", "distance"]}
    knn_power = {"knn__p": [1, 2]}
    knn_feature_selection = {"feature_selection__k": [5, 7, 9, 10, 12, 14, 16, 18]}

    tuning_params_1 = (
            column_transformation_tune
            | over_sampling_tuning
            | over_sampling__sampling_strategy_tuning
            | under_sampling_tuning
            | knn__n_neighbors_tuning
            | knn__weights
            | knn_power
            | knn_feature_selection
    )

    tuning_params_2 = (
            column_transformation_tune
            | under_sampling_tuning
            | knn__n_neighbors_tuning
            | knn__weights
            | knn_power
            | knn_feature_selection
    )

    best_model = (
        TuneHyperParams()
        .random_grid_search(
            column_transforms_only_model, [tuning_params_1, tuning_params_2]
        )
        .fit_model(data.TRAINING.predictors, data.TRAINING.outcome)
        .get_best_model()
    )

    final_model_knn = FinalModelPerformance(model=best_model, data=data)
    final_model_knn.get_cross_validation_results()
    final_model_knn.get_final_model_performance()

    dump(final_model_knn.model, Locations.models_exports / ModelFileNames.knn)
