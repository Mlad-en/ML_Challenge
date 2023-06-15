from utils import (
    FinalModelPerformance,
    TransactionDataset,
    TuneHyperParams,
)

from constants import (
    Columns,
    ModelConstants,
    ResamplingStrategy,
    ModelFileNames,
    Locations,
)

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    Normalizer,
    PowerTransformer,
    RobustScaler,
)
from sklearn.compose import make_column_transformer, make_column_selector

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks, NearMiss

from joblib import dump


if __name__ == "__main__":
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

    regression_model = LogisticRegression(
        random_state=ModelConstants.RANDOM_STATE,
        max_iter=ModelConstants.MAX_ITERATIONS * 10,
        warm_start=True,
    )

    smote_oversampling = SMOTE(random_state=ModelConstants.RANDOM_STATE)
    adasyn_oversampling = ADASYN(random_state=ModelConstants.RANDOM_STATE)
    tomek_under_sampling = TomekLinks()
    near_miss_under_sampling = NearMiss()

    column_transforms_only_model = Pipeline(
        [
            ("column_transformation", log_and_ordinal_transforms),
            ("over_sampling", smote_oversampling),
            ("under_sampling", tomek_under_sampling),
            ("logistic_regression", regression_model),
        ]
    )

    tune_logistic_regression_penalty = {"logistic_regression__penalty": ["l1", "l2"]}
    tune_logistic_regression_solver = {
        "logistic_regression__solver": ["saga", "liblinear"]
    }
    tune_logistic_regression_C = {
        "logistic_regression__C": [0.1, 0.5, 1, 1.5, 2, 2.5, 3]
    }
    tune_logistic_regression_class_weight = {
        "logistic_regression__class_weight": [
            None,
            "balanced",
            {1: 0.55, 0: 0.45},
            {1: 0.6, 0: 0.4},
            {1: 0.65, 0: 0.35},
            {1: 0.7, 0: 0.3},
            {1: 0.75, 0: 0.25},
            {1: 0.8, 0: 0.2},
        ]
    }

    tune_column_transformation = {
        "column_transformation": [
            log_and_ordinal_transforms,
            norm_and_one_hot_transforms,
            power_transforms,
            robust_transformation,
        ],
    }

    tune_over_sampling = {
        "over_sampling": [
            smote_oversampling,
            adasyn_oversampling,
        ]
    }

    tune_over_sampling__sampling_strategy = {
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
    tune_under_sampling = {
        "under_sampling": [
            near_miss_under_sampling,
            tomek_under_sampling,
            "passthrough",
        ],
    }

    tuning_params_1 = (
        tune_column_transformation
        | tune_logistic_regression_class_weight
        | tune_logistic_regression_penalty
        | tune_logistic_regression_solver
        | tune_logistic_regression_C
        | tune_over_sampling
        | tune_over_sampling__sampling_strategy
        | tune_under_sampling
    )

    tuning_params_2 = (
        tune_column_transformation
        | tune_logistic_regression_class_weight
        | tune_logistic_regression_penalty
        | tune_logistic_regression_solver
        | tune_logistic_regression_C
        | tune_under_sampling
    )

    best_model = (
        TuneHyperParams()
        .random_grid_search(
            column_transforms_only_model, [tuning_params_1, tuning_params_2]
        )
        .fit_model(data.TRAINING.predictors, data.TRAINING.outcome)
        .get_best_model()
    )

    final_model_performance_lr = FinalModelPerformance(model=best_model, data=data)

    final_model_performance_lr.get_cross_validation_results()
    final_model_performance_lr.get_final_model_performance()

    dump(
        final_model_performance_lr.model,
        Locations.models_exports / ModelFileNames.logistic_regression,
    )
