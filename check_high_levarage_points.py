from yellowbrick.regressor import cooks_distance
from utils import TransactionDataset
import matplotlib.pyplot as plt
from constants import Columns
from pathlib import Path

path = Path(".")
eda_results = path / "eda_results"


if __name__ == "__main__":
    data = TransactionDataset().get_training_test_split()

    data.TRAINING.predictors.pop(Columns.CUSTOMER_TYPE)
    data.TRAINING.predictors.pop(Columns.SPECIFIC_HOLIDAY)
    data.TRAINING.predictors.pop(Columns.WEEKDAY)

    cooks_distance(
        data.TRAINING.predictors, data.TRAINING.outcome, draw_threshold=True, show=False
    )
    plt.savefig(eda_results / "features_cooks_distance.png")
