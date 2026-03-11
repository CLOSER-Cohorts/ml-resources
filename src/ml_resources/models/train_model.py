from sklearn.linear_model import LogisticRegression
import numpy as np

def train_model(data_for_model, selected_input_features=None, prediction_model=LogisticRegression()):
    input_feature_list=[]
    for input_feature in selected_input_features:
        input_feature_list.append(np.vstack(data_for_model['X_train'][input_feature]))
    X = np.hstack(
        input_feature_list
    )
    if selected_input_features is None:
        prediction_model.fit(X, data_for_model['y_train'].squeeze())
    else:
        prediction_model.fit(X, data_for_model['y_train'].squeeze())
    return prediction_model