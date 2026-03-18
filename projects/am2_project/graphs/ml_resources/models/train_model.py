from sklearn.linear_model import LogisticRegression

def train_model(data_for_model, selected_input_features=None, prediction_model=LogisticRegression()):
    if selected_input_features is None:
        prediction_model.fit(list(data_for_model['X_train'].values), data_for_model['y_train'].squeeze())
    else:
        prediction_model.fit(list(data_for_model['X_train'][selected_input_features].values), data_for_model['y_train'].squeeze())
    return prediction_model