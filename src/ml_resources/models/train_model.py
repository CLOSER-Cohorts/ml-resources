from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
import numpy as np

param_dist = {
    "C": loguniform(7.5e1, 1e2),              # regularisation strength (log scale)
    "penalty": ["l2"],   # type of regularisation
    #"l1_ratio": [0, 1],
    #"solver": ["saga"],         # solvers that support L1 / elasticnet
    #"l1_ratio": loguniform(1e-3, 1),         # only used if penalty='elasticnet'
    "class_weight": [None, "balanced"]
}


def train_model(data_for_model, selected_input_features=None, 
    prediction_model=LogisticRegression(max_iter=1000)):
    input_feature_list=[]
    for input_feature in selected_input_features:
        input_feature_list.append(np.vstack(data_for_model['X_train'][input_feature]))
    X_train = np.hstack(
        input_feature_list
    )
   # prediction_model2=LogisticRegression(max_iter=1000)
    
    prediction_model2=LogisticRegression(max_iter=1000,
        class_weight='balanced',
        penalty='l2',
        C=86.26)
    
    """
    model = LogisticRegression(max_iter=1000)
    prediction_model2 = RandomizedSearchCV(
        prediction_model,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1_macro",     # or "accuracy", "roc_auc", etc.
        #cv=1,
        verbose=3,
        n_jobs=-1,
        random_state=42
        )     
    prediction_model2.fit(X_train, data_for_model['y_train'].squeeze())
    """
    prediction_model2.fit(X_train,
        data_for_model['y_train'].squeeze())
    """
    if selected_input_features is None:
        prediction_model.fit(X_train, data_for_model['y_train'].squeeze())
    else:
        prediction_model.fit(X_train, data_for_model['y_train'].squeeze())
    """
    return prediction_model2