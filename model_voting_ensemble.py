#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import load
from sklearn.ensemble import VotingClassifier
from joblib import dump

from utils import TransactionDataset, FinalModelPerformance, TuneHyperParams


# In[2]:


lr_model = load("./models_exports/logistic_regression_classifier.joblib")
knn_model = load("./models_exports/knn_classifier.joblib")
rf_model = load("./models_exports/random_forest_classifier.joblib")
nb_model = load("./models_exports/naive_bayes_classifier.joblib")
svc_model = load("./models_exports/support_vector_classifier.joblib")


# In[3]:


data = TransactionDataset().get_training_test_split()


# In[4]:


voting_classifier = VotingClassifier(
    estimators=[
        ("lr_model", lr_model),
        ("knn_model", knn_model),
        ("rf_model", rf_model),
        ("nb_model", nb_model),
        ("svc_model", svc_model)
    ],
)


# In[5]:


final_model_performance = FinalModelPerformance(
    model=voting_classifier,
    data=data
)


# In[6]:


final_model_performance.get_cross_validation_results()


# In[7]:


final_model_performance.get_final_model_performance()


# In[9]:


dump(final_model_performance.model, "./models_exports/voting_ensemble.joblib")


# In[ ]:




