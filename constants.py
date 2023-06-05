from enum import Enum, auto


class Columns:

    TRANSACTION = "Transaction"
    CUSTOMER_TYPE = "Customer_Type"
    SYSTEM_F1 = "SystemF1"
    SYSTEM_F2 = "SystemF2"
    SYSTEM_F3 = "SystemF3"
    SYSTEM_F4 = "SystemF4"
    SYSTEM_F5 = "SystemF5"
    ACCOUNT_PAGE = "Account_Page"
    ACCOUNT_PAGE_TIME = "Account_Page_Time"
    INFO_PAGE = "Info_Page"
    INFO_PAGE_TIME = "Info_Page_Time"
    PRODUCT_PAGE = "ProductPage"
    PRODUCT_PAGE_TIME = "ProductPage_Time"
    MONTH = "Month"
    WEEKDAY = "Weekday"
    SPECIFIC_HOLIDAY = "SpecificHoliday"
    GOOGLE_ANALYTICS_BR = "GoogleAnalytics_BR"
    GOOGLE_ANALYTICS_ER = "GoogleAnalytics_ER"
    GOOGLE_ANALYTICS_PV = "GoogleAnalytics_PV"
    AD_CAMPAIGN_1 = "Ad_Campaign_1"
    AD_CAMPAIGN_2 = "Ad_Campaign2"
    AD_CAMPAIGN_3 = "Ad_Campaign3"


class ModelConstants:
    
    MAX_ITERATIONS = 1000
    RANDOM_STATE = 123
    F1_SCORE = "f1"
    ACCURACY_SCORE = "accuracy"
    BALANCED_ACCURACY_SCORE = "balanced_accuracy"
    CROSS_VALIDATIONS = 5
    N_JOBS = 5


class ResamplingStrategy:

    AUTO = "auto"
    MINORITY_ONLY = "minority"
    ALL = "all"
