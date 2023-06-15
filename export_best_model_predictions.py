import joblib
from constants import Locations, ModelFileNames
from utils import TransactionDataset
import numpy as np

if __name__ == "__main__":
    rf_model = joblib.load(Locations.models_exports / ModelFileNames.random_forest)

    data = TransactionDataset().get_training_test_split()

    predictions = rf_model.predict(data.TESTING.predictors)
    np.savetxt(
        Locations.models_exports / "rf_predictions.csv",
        predictions,
        delimiter=",",
        header="RF Test Predictions",
    )
