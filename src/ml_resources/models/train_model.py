from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint
import numpy as np
from src.dataframe_utility import convert_df_to_ndarray
from src.ml_resources.models.predict_model import calculate_accuracy
    
param_dist = {
    "C": loguniform(7.5e1, 1e2),              # regularisation strength (log scale)
    "penalty": ["l2"],   # type of regularisation
    #"l1_ratio": [0, 1],
    #"solver": ["saga"],         # solvers that support L1 / elasticnet
    #"l1_ratio": loguniform(1e-3, 1),         # only used if penalty='elasticnet'
    "class_weight": [None, "balanced"]
}
"""

path = clf.cost_complexity_pruning_path(lr_model_data['X_train'], lr_model_data['y_train'])
ccp_alphas = path.ccp_alphas
ccp_alphas = np.clip(ccp_alphas, 0, None)
impurities = path.impurities
dtcs = []
train_scores = []
test_scores = []    
n_leaves = []

# Train a tree for each alpha
count=0
for alpha in ccp_alphas:
    print(count)
    count=count+1
    dtc_ = DecisionTreeClassifier(random_state=136, ccp_alpha=float(alpha))
    dtc_.fit(convert_df_to_ndarray(lr_model_data['X_train'][0:3000]), lr_model_data['y_train'][0:3000])
    dtcs.append(dtc_)
    train_scores.append(dtc_.score(convert_df_to_ndarray(lr_model_data['X_train'][0:3000]), lr_model_data['y_train'][0:3000]))
    test_scores.append(dtc_.score(convert_df_to_ndarray(lr_model_data['X_test'][0:3000]), lr_model_data['y_test'][0:3000]))
    n_leaves.append(dtc_.get_n_leaves())


param_dist = {
    "max_depth": [50,100,200,500],
    "splitter": ["best", "random"],
    "class_weight": ["balanced", None],
    "max_features": [None],
    "min_samples_split": [0,10,30,50]
}

param_dist = {
    "max_depth": [100],
    "splitter": ["best", "random"],
    "class_weight": ["balanced", None],
    "max_features": [None],
    "min_samples_split": [30]
}

prediction_model2 = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=5,
        scoring="f1_macro",     # or "accuracy", "roc_auc", etc.
        #cv=1,
        verbose=3,
        n_jobs=-1,
        random_state=42
        )     

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(DecisionTreeClassifier(), param_dist, cv=3, verbose=3, n_jobs=1)
grid_search.fit(convert_df_to_ndarray(lr_model_data['X_train']), lr_model_data['y_train'])
y_pred=grid_search.predict(convert_df_to_ndarray(lr_model_data['X_test']))   
report = classification_report(lr_model_data['y_test'].values, y_pred)




    "C": loguniform(7.5e1, 1e2),              # regularisation strength (log scale)
    "penalty": ["l2"],   # type of regularisation
    #"l1_ratio": [0, 1],
    #"solver": ["saga"],         # solvers that support L1 / elasticnet
    #"l1_ratio": loguniform(1e-3, 1),         # only used if penalty='elasticnet'
    "class_weight": [None, "balanced"]
}
"""

def train_model(data_for_model, 
    feature_columns,
    prediction_model=LogisticRegression(max_iter=1000)):
    X_train=convert_df_to_ndarray(data_for_model['X_train'][feature_columns], input_features=feature_columns)
    #prediction_model2=LogisticRegression(max_iter=1000)
    param_grid={
        'C': [0.1, 1, 10, 15, 20, 30, 100],
        'max_iter': [5000],
        'penalty': ['l2'],
        'class_weight': ["balanced"]
        }
    max_accuracy=0
    prediction_model=None
    for params in ParameterSampler(param_grid, n_iter=100):
        trained_model = LogisticRegression(**params, )
        trained_model.fit(X_train, data_for_model['y_train'])
        X_validation=convert_df_to_ndarray(data_for_model['X_test'], input_features=feature_columns)
        predictions_with_probabilities=trained_model.predict_proba(X_validation)
        y_validation=data_for_model['y_test']
        prediction_results=calculate_accuracy(trained_model,
            predictions_with_probabilities,
            X_validation,
            y_validation.tolist(),
            N=5)
        if prediction_results['Accuracy']>max_accuracy:
            max_accuracy=prediction_results['Accuracy']	
            print(f"Best so far: {max_accuracy}")
            print(params)
            prediction_model=trained_model
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
    """
    if isinstance(prediction_model, XGBClassifier):
        le = LabelEncoder()
        y_train = le.fit_transform(data_for_model['y_train'])
    else:
        y_train=data_for_model['y_train']
    """
    return prediction_model
