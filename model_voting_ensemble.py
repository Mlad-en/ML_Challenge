from joblib import load
from sklearn.ensemble import VotingClassifier
from joblib import dump

from utils import TransactionDataset, FinalModelPerformance, TuneHyperParams
from constants import ModelFileNames, Locations


if __name__ == '__main__':

    lr_model = load(Locations.models_exports / ModelFileNames.logistic_regression)
    knn_model = load(Locations.models_exports / ModelFileNames.knn)
    rf_model = load(Locations.models_exports / ModelFileNames.random_forest)
    nb_model = load(Locations.models_exports / ModelFileNames.naive_bayes)
    svc_model = load(Locations.models_exports / ModelFileNames.svc)

    data = TransactionDataset().get_training_test_split()

    voting_classifier = VotingClassifier(
        estimators=[
            ("lr_model", lr_model),
            ("knn_model", knn_model),
            ("rf_model", rf_model),
            ("nb_model", nb_model),
            ("svc_model", svc_model),
        ],
    )

    final_model_performance = FinalModelPerformance(model=voting_classifier, data=data)

    final_model_performance.get_cross_validation_results()
    final_model_performance.get_final_model_performance()

    dump(final_model_performance.model, Locations.models_exports / ModelFileNames.voting_ensemble)
