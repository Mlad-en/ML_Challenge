import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from constants import Columns, Locations
from utils import TransactionDataset, format_output

from logging import getLogger, basicConfig, INFO


sns.set_style("dark")

basicConfig(
    level=INFO,
    filename=Locations.eda_results / "features.log",
    filemode="a",
    format="{message}",
    style="{",
)

logger = getLogger(__name__)


if __name__ == "__main__":
    df = TransactionDataset().data

    logger.info(format_output(df.isnull().sum(), "Are there any null values"))
    logger.info(format_output(df.Transaction.value_counts(), "Transaction Counts"))

    trans_pcnt = df.Transaction.value_counts() / len(df)
    logger.info(format_output(trans_pcnt, "Transactions as Percentages"))

    logger.info(
        format_output(df[Columns.CUSTOMER_TYPE].unique(), "Unique Customer Types")
    )
    logger.info(
        format_output(
            df[[Columns.CUSTOMER_TYPE]].value_counts(), "Counts per Customer Type"
        )
    )

    cross_tab_trans_cust = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.CUSTOMER_TYPE], margins=True
    )
    cross_tab_trans_cust_title = (
        "Customer Counts Cross Tabulated by Transaction Success"
    )
    logger.info(format_output(cross_tab_trans_cust, cross_tab_trans_cust_title))

    pnct_cross_tab_trans_cust = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.CUSTOMER_TYPE], normalize=True
    )
    pnct_cross_tab_trans_cust_title = (
        "Customer Counts Cross Tabulated by Transaction Success"
    )
    logger.info(
        format_output(pnct_cross_tab_trans_cust, pnct_cross_tab_trans_cust_title)
    )

    system_variables = [
        Columns.SYSTEM_F1,
        Columns.SYSTEM_F2,
        Columns.SYSTEM_F3,
        Columns.SYSTEM_F4,
        Columns.SYSTEM_F5,
    ]

    logger.info(
        format_output(df[system_variables].describe(), "Describe System F1-F5 Features")
    )
    logger.info(
        format_output(df[system_variables].skew(), "Show Skew For F1-F5 Features")
    )

    box_plot = sns.boxplot(
        data=df[system_variables].melt(var_name="system", value_name="score"),
        x="system",
        y="score",
    )
    plt.savefig(Locations.eda_results / "systems_boxplot.png")

    axis_indices = [(row, col) for row in range(0, 3) for col in range(0, 2)]
    color_indices = ["maroon", "rosybrown", "coral", "salmon", "orangered"]

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.delaxes(axs.flatten()[-1])

    for index in range(0, len(axis_indices) - 1):
        sns.histplot(
            data=df,
            x=system_variables[index],
            kde=True,
            color=color_indices[index],
            ax=axs[axis_indices[index]],
        )

    plt.savefig(Locations.eda_results / "systems_distributions.png")

    axis_indices = [(row, col) for row in range(0, 3) for col in range(0, 2)]
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    fig.delaxes(axs.flatten()[-1])
    for index in range(0, len(axis_indices) - 1):
        sns.kdeplot(
            data=df,
            x=system_variables[index],
            hue=df[Columns.TRANSACTION],
            ax=axs[axis_indices[index]],
        )

    plt.savefig(Locations.eda_results / "systems_kde_distributions_by_transaction.png")

    pages = [Columns.ACCOUNT_PAGE, Columns.INFO_PAGE, Columns.PRODUCT_PAGE]

    page_times = [
        Columns.ACCOUNT_PAGE_TIME,
        Columns.INFO_PAGE_TIME,
        Columns.PRODUCT_PAGE_TIME,
    ]

    logger.info(format_output(df[pages].describe(), "Describe Page Features"))
    logger.info(format_output(df[pages].skew(), "Show Skew For Page Features"))

    logger.info(format_output(df[page_times].describe(), "Describe Page Time Features"))
    logger.info(
        format_output(df[page_times].skew(), "Show Skew For Page Time Features")
    )

    sns.boxplot(
        data=df[pages].melt(var_name="page", value_name="score"), x="page", y="score"
    )
    plt.savefig(Locations.eda_results / "pages_boxplot.png")

    sns.boxplot(
        data=df[page_times].melt(var_name="page_time", value_name="score"),
        x="page_time",
        y="score",
    )
    plt.savefig(Locations.eda_results / "page_times_boxplot.png")

    days = [Columns.MONTH, Columns.SPECIFIC_HOLIDAY, Columns.WEEKDAY]
    logger.info(format_output(df[days].describe(), "Describe Days Features"))

    cross_tab_trans_month = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.MONTH], margins=True
    )
    logger.info(format_output(cross_tab_trans_month, "Cross Tab Transactions By Month"))
    pcnt_cross_tab_trans_month = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.MONTH], normalize=True
    ).round(2)
    logger.info(
        format_output(
            pcnt_cross_tab_trans_month, "Cross Tab Transactions By Month (Percentage)"
        )
    )

    logger.info(format_output(df[days].skew(), "Show Skew For Days Features"))
    logger.info(
        format_output(
            df[Columns.WEEKDAY].value_counts(),
            "Show How Many times Different weekdays occur",
        )
    )

    sns.histplot(data=df, x=df[Columns.MONTH], hue=df[Columns.TRANSACTION], kde=True)
    plt.savefig(Locations.eda_results / "hist_kde_Monthly_transactions.png")

    sns.histplot(
        data=df,
        x=df[Columns.SPECIFIC_HOLIDAY].astype(np.float64),
        hue=df[Columns.TRANSACTION],
        kde=True,
    )
    plt.savefig(Locations.eda_results / "hist_kde_specific_holiday_transactions.png")

    sns.histplot(
        data=df,
        x=df[df[Columns.SPECIFIC_HOLIDAY].astype(np.float64) > 0][
            Columns.SPECIFIC_HOLIDAY
        ].astype(np.float64),
        hue=df[Columns.TRANSACTION],
        kde=True,
    )
    plt.savefig(Locations.eda_results / "hist_kde_on_holiday_transactions.png")

    cross_tab_trans_holiday = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.SPECIFIC_HOLIDAY], margins=True
    )
    logger.info(
        format_output(cross_tab_trans_holiday, "Cross Tab Transaction by Holiday")
    )

    pnct_cross_tab_trans_holiday = pd.crosstab(
        df[Columns.TRANSACTION], df[Columns.SPECIFIC_HOLIDAY], normalize=True
    ).round(3)

    logger.info(
        format_output(
            pnct_cross_tab_trans_holiday,
            "Cross Tab Transaction by Holiday (Percentage)",
        )
    )

    ad_campaigns = [
        Columns.AD_CAMPAIGN_1,
        Columns.AD_CAMPAIGN_2,
        Columns.AD_CAMPAIGN_3,
    ]

    google_analytics = [
        Columns.GOOGLE_ANALYTICS_BR,
        Columns.GOOGLE_ANALYTICS_ER,
        Columns.GOOGLE_ANALYTICS_PV,
    ]

    logger.info(format_output(df[ad_campaigns].describe(), "Describe Ad Campaigns"))

    logger.info(
        format_output(
            df[ad_campaigns].value_counts(),
            "Count How Many Records were Collected During Ad Campaigns",
        )
    )

    cross_tab_ad1_transaction = pd.crosstab(
        df[Columns.AD_CAMPAIGN_1], df[Columns.TRANSACTION], margins=True
    )
    logger.info(
        format_output(
            cross_tab_ad1_transaction, "Cross Tab Ad Campaign 1 with Transactions"
        )
    )

    cross_tab_ad2_transaction = pd.crosstab(
        df[Columns.AD_CAMPAIGN_2], df[Columns.TRANSACTION], margins=True
    )
    logger.info(
        format_output(
            cross_tab_ad2_transaction, "Cross Tab Ad Campaign 2 with Transactions"
        )
    )

    cross_tab_ad3_transaction = pd.crosstab(
        df[Columns.AD_CAMPAIGN_3], df[Columns.TRANSACTION], margins=True
    )
    logger.info(
        format_output(
            cross_tab_ad3_transaction, "Cross Tab Ad Campaign 3 with Transactions"
        )
    )

    logger.info(
        format_output(
            df[google_analytics].describe(), "Describe Google Analytics Features"
        )
    )
    logger.info(
        format_output(df[google_analytics].skew(), "Show Google Analytics Feature Skew")
    )

    sns.boxplot(
        data=df[google_analytics[0:2]].melt(var_name="analytics", value_name="score"),
        x="analytics",
        y="score",
    )
    plt.savefig(Locations.eda_results / "analytics_boxplot.png")

    sns.boxplot(data=df[Columns.GOOGLE_ANALYTICS_PV])
    fig, axs = plt.subplots(3, figsize=(7, 7))
    for index in range(0, 3):
        sns.histplot(
            data=df,
            x=google_analytics[index],
            kde=True,
            color=color_indices[index],
            ax=axs[index],
        )
    plt.savefig(Locations.eda_results / "google_analytics_distributions.png")

    fig, axs = plt.subplots(3, figsize=(10, 10))
    for index in range(0, 3):
        sns.kdeplot(
            data=df,
            x=google_analytics[index],
            hue=df[Columns.TRANSACTION],
            ax=axs[index],
        )
    plt.savefig(
        Locations.eda_results / "google_analytics_per_transaction_distributions.png"
    )

    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(method="pearson").round(3), annot=True)
    plt.savefig(Locations.eda_results / "feature_correlation.png")
