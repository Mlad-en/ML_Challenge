import logging
from logging import basicConfig, INFO
from pathlib import Path

from utils import TransactionDataset, format_output
from constants import Columns

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


path = Path(".")
eda_results = path / "eda_results"

basicConfig(
    level=INFO,
    filename=eda_results / "vif.log",
    filemode='w',
    format="{message}",
    style="{"
)


if __name__ == '__main__':

    data = TransactionDataset()
    data.data.pop(Columns.TRANSACTION)

    data.data = pd.get_dummies(data.data, columns=[Columns.CUSTOMER_TYPE, Columns.SPECIFIC_HOLIDAY])
    data.data = data.data.astype("float")

    vif_data = pd.DataFrame()
    vif_data["feature"] = data.data.columns

    vif_data["VIF"] = [variance_inflation_factor(data.data.values, i)
                       for i in range(len(data.data.columns))]

    logging.info(format_output(vif_data, "Variance Inflation Factor"))
