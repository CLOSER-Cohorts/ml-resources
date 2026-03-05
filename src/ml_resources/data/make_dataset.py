from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def create_dataset(identifiers, 
    input_features,
    input_feature_name,
    target_name,
    targets):
    X=pd.DataFrame({}, columns=[input_feature_name])
    y=pd.DataFrame({}, columns=[target_name])
    for identifier in identifiers:
        X.loc[identifier] = [input_features[identifier]]
        y.loc[identifier] = [targets[identifier]]
    return { "InputFeatures": X, "Targets": y}
    
def add_input_feature_to_dataset(identifiers, 
    input_features, 
    new_input_feature_name, 
    dataset):
    updated_input_features=dataset['InputFeatures'].loc[identifiers]
    updated_input_features[new_input_feature_name] = {k: input_features[k] for k in identifiers if k in input_features}
    updated_targets=dataset['Targets'].loc[identifiers]
    return { "InputFeatures": updated_input_features, "Targets": updated_targets}

def create_model_data_object(X_train, X_test, y_train, y_test):
    dataForTrainingAndTest = {}
    dataForTrainingAndTest['X_train'] = X_train
    dataForTrainingAndTest['X_test'] = X_test
    dataForTrainingAndTest['y_train'] = y_train
    dataForTrainingAndTest['y_test'] = y_test
    return dataForTrainingAndTest
