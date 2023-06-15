import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from joblib import dump


from constants import Columns, ModelConstants, ModelFileNames, Locations
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

    sv_classifier = SVC(random_state=ModelConstants.RANDOM_STATE)

    svc_pipeline = Pipeline(
        [
            ("column_transformation", column_transformation_1),
            ("data_sampling", smote_oversampling),
            ("svc_model", sv_classifier),
        ]
    )

    params_to_tune = {
        "column_transformation": [column_transformation_1],
        "data_sampling": [smote_oversampling, adasyn_oversampling, "passthrough"],
        "svc_model__C": [0.1, 1, 10, 100],
        "svc_model__kernel": ["linear"],
    }

    best_model_svc = (
        TuneHyperParams()
        .random_grid_search(svc_pipeline, params_to_tune)
        .fit_model(data.TRAINING.predictors, data.TRAINING.outcome)
        .get_best_model()
    )

    final_model_evaluation_svc = FinalModelPerformance(model=best_model_svc, data=data)

    final_model_evaluation_svc.get_cross_validation_results()

    final_model_evaluation_svc.get_final_model_performance()

    dump(
        final_model_evaluation_svc.model,
        Locations.models_exports / ModelFileNames.svc
    )
