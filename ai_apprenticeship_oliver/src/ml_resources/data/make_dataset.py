from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import copy

def convert_dictionary_to_dataframe(dict_obj):
    data = []
    count=0
    for study_agency_id, items in dict_obj.items():
        for item_id, item_data in items.items():
            print(count)
            count=count+1
            row = {"id": item_id}
            for key, value in item_data.items():
                if key == "ItemCategories":
                    row[key] = " ".join(sorted(value)).lower()
                else:
                    row[key] = value
            data.append(row)
    return pd.DataFrame(data).set_index("id")

def update_dataset(study_agency_id,
    identifiers,
    input_features,
    input_feature_name,
    target_name,
    targets,
    dataset={}):
    """Updates an existing dataset specified in the dataset keyword input argument, or 
    creates a new dataset if an existing dataset is not specified. 
    IS THIS FN NEEDED NOW?"""
    if study_agency_id not in dataset:
        X=pd.DataFrame({}, columns=[input_feature_name])
        y=pd.DataFrame({}, columns=[target_name])
        dataset[study_agency_id]={"InputFeatures": X, "Targets": y}
    for identifier in identifiers:
        if identifier not in dataset[study_agency_id].keys():
            dataset[study_agency_id]["InputFeatures"].loc[identifier] = [input_features[identifier]]
            dataset[study_agency_id]["Targets"].loc[identifier] = [targets[identifier]]
    return dataset

def add_input_feature_to_dataset(study_agency_id,
    identifiers,
    input_features,
    new_input_feature_name,
    dataset):
    updated_input_features=dataset[study_agency_id]['InputFeatures']#.loc[identifiers]
    updated_input_features[new_input_feature_name] = {k: input_features[study_agency_id][k] for k in identifiers if k in input_features[study_agency_id]}
    updated_targets = dataset[study_agency_id]['Targets'].loc[identifiers]
    dataset[study_agency_id]['InputFeatures'] = updated_input_features
    dataset[study_agency_id]['Targets'] = updated_targets

def create_model_data_object(X_train, X_test, y_train, y_test):
    dataForTrainingAndTest = {}
    dataForTrainingAndTest['X_train'] = X_train
    dataForTrainingAndTest['X_test'] = X_test
    dataForTrainingAndTest['y_train'] = y_train
    dataForTrainingAndTest['y_test'] = y_test
    return dataForTrainingAndTest

def transform_input_feature_categories(data, study_agency_id, categories):
    dataCopy=copy.deepcopy(data)
    dataCopy[study_agency_id]['InputFeatures'][categories] = dataCopy[study_agency_id]['InputFeatures'][categories].apply(lambda x: " ".join(sorted(x)).lower())
    return dataCopy

def filter_values_by_length(data_values,
    filter_attribute,
    length,
    include_zero_length_items=False,
    filter_type='greater_than'):
    filtered_values=copy.deepcopy(data_values)
    for study in data_values.keys():
        #for item in data_values.keys():
            if filter_type == 'greater_than':
                filtered_values[study] = {k: v for k, v in data_values[study].items() 
                    if (filter_attribute in v.keys() and
                        (len(v[filter_attribute]) > length or 
                        (len(v[filter_attribute])==0 and include_zero_length_items)))}
            elif filter_type == 'less_than':
                filtered_values[study] = {k: v for k, v in data_values[study].items() 
                    if (filter_attribute in v.keys() and
                        (len(v[filter_attribute]) < length or
                        (len(v[filter_attribute])==0 and include_zero_length_items)))}
            else:
                print(f"Unknown filter type {filter_type}.")
    return filtered_values

def check_for_newly_available_data_am1(ids_from_repository_for_dataset,
    current_project_data_ids):
    newly_available_data_identifiers=[]
    if sorted(ids_from_repository_for_dataset)==sorted(current_project_data_ids):
        print("No new data available for model")
    else:
        print("The following new items are available to use for training/testing the model:")
        newly_available_data_identifiers=[x for x in ids_from_repository_for_dataset if x not in current_project_data_ids]
        print(newly_available_data_identifiers)
    return newly_available_data_identifiers

#a=filter_values_by_length(test_data, "QuestionCategories", 3)
#{k: v for k, v in test_data['uk.iser.ukhls'].items() if len(v["QuestionCategories"].items()) > length}
"""
    return categories["QuestionCategories"].apply(
        lambda cats: " ".join(sorted(cats)).lower()
    ).to_frame()

    return " ".join(sorted(categories)).lower()

category_transformer = FunctionTransformer(transform_input_feature_categories, validate=False)
preprocessor = ColumnTransformer([
    ("Categories", category_transformer, ["QuestionCategories"])
],
)
pipeline = Pipeline([("transform_categories2", preprocessor)])
data = pd.DataFrame(X_train, columns=['QuestionCategories'])
transformed_categories = pd.DataFrame({"embeddings": list(pipeline.fit_transform(data))})

def filter_rows_from_dataset(dataset, column, column_value_to_filter):
"""
