import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from joblib import dump


from constants import Columns, ModelConstants, Locations, ModelFileNames
from utils import TuneHyperParams, TransactionDataset, FinalModelPerformance


if __name__ == "__main__":
    data = TransactionDataset().get_training_test_split()

    drop_columns = [
        Columns.INFO_PAGE_TIME,
        Columns.ACCOUNT_PAGE_TIME,
        Columns.PRODUCT_PAGE_TIME,
        Columns.AD_CAMPAIGN_1,
        Columns.AD_CAMPAIGN_2,
        Columns.AD_CAMPAIGN_3,
    ]

    column_transformation_1 = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (OrdinalEncoder(), [Columns.SPECIFIC_HOLIDAY]),
        (FunctionTransformer(np.log1p), make_column_selector(dtype_include="number")),
        ("drop", drop_columns),
        remainder="passthrough",
    )

    column_transformation_2 = make_column_transformer(
        (OneHotEncoder(), [Columns.CUSTOMER_TYPE]),
        (OrdinalEncoder(), [Columns.SPECIFIC_HOLIDAY]),
        remainder="passthrough",
    )

    smote_oversampling = SMOTE(random_state=ModelConstants.RANDOM_STATE)
    adasyn_oversampling = ADASYN(random_state=ModelConstants.RANDOM_STATE)

    nb_model = BernoulliNB(force_alpha=True)

    principal_components = PCA()

    nb_model_pipeline = Pipeline(
        [
            ("column_transformations", column_transformation_1),
            ("data_sampling", smote_oversampling),
            ("principal_components", principal_components),
            ("nb_model", nb_model),
        ]
    )

    params_to_tune = {
        "column_transformations": [column_transformation_1, column_transformation_2],
        "data_sampling": [smote_oversampling, adasyn_oversampling, "passthrough"],
        "principal_components__n_components": [3, 5, 7, 9, 11, 13],
        "nb_model__alpha": [0, 0.7, 0.9, 1],
        "nb_model__class_prior": [
            [0.7, 0.25],
            [0.6, 0.3],
            [0.6, 0.35],
            [0.7, 0.3],
            [0.55, 0.35],
            [0.55, 0.45],
        ],
    }

    best_model_nb = (
        TuneHyperParams()
        .full_grid_search(nb_model_pipeline, params_to_tune)
        .fit_model(data.TRAINING.predictors, data.TRAINING.outcome)
        .get_best_model()
    )

    final_model_evaluation_nb = FinalModelPerformance(model=best_model_nb, data=data)

    final_model_evaluation_nb.get_cross_validation_results()
    final_model_evaluation_nb.get_final_model_performance()

    dump(
        final_model_evaluation_nb.model,
        Locations.models_exports / ModelFileNames.naive_bayes,
    )
