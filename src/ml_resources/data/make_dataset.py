from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

def update_dataset(study_agency_id,
    identifiers,
    input_features,
    input_feature_name,
    target_name,
    targets,
    dataset={}):
    """Updates an existing dataset specified in the dataset keyword input argument, or 
    creates a new dataset if an existing dataset is not specified."""
    if study_agency_id not in dataset:
        X=pd.DataFrame({}, columns=[input_feature_name])
        y=pd.DataFrame({}, columns=[target_name])
        dataset[study_agency_id]={"InputFeatures": X, "Targets": y}
    for identifier in identifiers:
        if identifier not in dataset[study_agency_id].keys():
            dataset[study_agency_id]["InputFeatures"].loc[identifier] = [input_features[identifier]]
            dataset[study_agency_id]["Targets"].loc[identifier] = [targets[identifier]]
    return dataset

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
