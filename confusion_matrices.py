from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import load
from utils import TransactionDataset
from constants import Locations, ModelFileNames


if __name__ == "__main__":
    lg_mmodel = load(Locations.models_exports / ModelFileNames.logistic_regression)
    knn_model = load(Locations.models_exports / ModelFileNames.knn)
    nb_model = load(Locations.models_exports / ModelFileNames.naive_bayes)
    svc_model = load(Locations.models_exports / ModelFileNames.svc)
    rf_model = load(Locations.models_exports / ModelFileNames.random_forest)
    voting_model = load(Locations.models_exports / ModelFileNames.voting_ensemble)

    data = TransactionDataset().get_training_test_split()

    all_model_predictions = [
        (lg_mmodel.predict(data.TESTING.predictors), "Logistic Regression"),
        (knn_model.predict(data.TESTING.predictors), "K-Nearest Neighbours"),
        (nb_model.predict(data.TESTING.predictors), "Naive Bayes"),
        (svc_model.predict(data.TESTING.predictors), "Support Vector Classifier"),
        (rf_model.predict(data.TESTING.predictors), "Random Forest"),
        (voting_model.predict(data.TESTING.predictors), "Voting Ensemble"),
    ]

    colors = [
        ["Blues", "Greens"],
        ["Greens", "Blues"],
        ["Blues", "Greens"],
    ]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    for row in range(3):
        for col in range(2):
            model = all_model_predictions.pop(0)

            ConfusionMatrixDisplay.from_predictions(
                data.TESTING.outcome,
                model[0],
                cmap=colors[row][col],
                display_labels=["Non-Transaction", "Transaction"],
                ax=axs[row, col],
            )
            axs[row, col].set_title(model[1])

    fig.savefig(Locations.confusion_matrices / "confusion_matrices.png")
