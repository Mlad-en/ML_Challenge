import warnings

warnings.filterwarnings("ignore")

from joblib import load
from utils import TransactionDataset
import shap
import matplotlib.pyplot as plt
from constants import Locations, ModelFileNames


if __name__ == "__main__":
    model = load(Locations.models_exports / ModelFileNames.random_forest)
    data = TransactionDataset().get_training_test_split()

    x_train = model[0:1].fit_transform(data.TRAINING.predictors, data.TRAINING.outcome)

    explainer = shap.TreeExplainer(model[-1]).shap_values(x_train)

    feature_names = model[0:1].get_feature_names_out()
    feature_names = [feat.split("__")[-1] for feat in feature_names]

    shap_summary = shap.summary_plot(
        explainer, x_train, feature_names=feature_names, show=False
    )
    plt.savefig(Locations.explainable_ai / "shapley_values.png")
