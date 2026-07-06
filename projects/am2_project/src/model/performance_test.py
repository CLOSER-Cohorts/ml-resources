from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, recall_score
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from src.ml_resources import (
    read_dataset_from_file,
    get_max_file_version,
    get_max_folder_version)


def retrain_model(relationships_data_for_training_updated_model):
    folder = "./projects/am2_project/models/all_item_models"
    object_name = "all_item_models"
    file_version = get_max_file_version(Path(f"{folder}"), object_name)
    if file_version >0:
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        all_item_models=read_dataset_from_file(file_path)
    else:
        all_item_models={}
    model=all_item_models[item_type]['model']
    existing_training_data=all_item_models[item_type]['data']
    existing_training_data["ItemType"] = existing_training_data["ItemType"].astype("category").cat.codes
    existing_training_data=existing_training_data.drop(columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'Flagged'])
    existing_training_labels=all_item_models[item_type]['data']['Flagged']
    additional_training_data=relationships_data_for_training_updated_model[item_type].drop_duplicates().drop('Flagged', axis=1)
    additional_training_labels=relationships_data_for_training_updated_model[item_type].drop_duplicates()['Flagged']
    updated_training_data=pd.concat([existing_training_data, additional_training_data])
    updated_training_labels=pd.concat([existing_training_labels,additional_training_labels])
    updated_training_data['Flagged'] = updated_training_labels
    updated_training_data_unique=updated_training_data.drop_duplicates()
    updated_training_input=updated_training_data_unique.drop('Flagged', axis=1)
    updated_training_labels=updated_training_data_unique['Flagged']
    param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
    }
    recall_minus1 = make_scorer(
    recall_score,
    pos_label=-1
    )
    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        scoring=recall_minus1,
        cv=5,
        n_jobs=-1
    )
    grid.fit(updated_training_input, updated_training_labels)
    dtc=grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print("Best CV performance:", grid.best_score_)
    return dtc


def measure_prediction_accuracy(data_with_predictions):
    reports={}
    df_unique=data_with_predictions.drop_duplicates()
    df_y_true=data_with_predictions.copy()
    count=1
    for k,v in df_unique.iterrows():
        ground_truth=""
        print(v)
        print(k)
        while ground_truth not in ['y', 'n']:
                ground_truth=input(f"{count} of {len(df_unique)}, is this right? y/n ")
        if ground_truth=='n':
            mask = df_y_true.eq(v).all(axis=1)
            print(mask)
            if v['Flagged']==1:
               df_y_true.loc[mask, 'Flagged']=-1    
            else:
               df_y_true.loc[mask, 'Flagged']=1
        ground_truth=""
        count = count + 1
    report = classification_report(df_y_true['Flagged'],
            data_with_predictions['Flagged'],
            output_dict=True
            )    
    return {"report": report, "ground_truth": df_y_true['Flagged']}


# Read in the input features and predictions from the production environment
folder = "./projects/am2_project/data/pending_training_data/new_items"
object_name = "new_items"
current_file_version=1
max_file_version = get_max_file_version(Path(f"{folder}"), object_name)
dicts=[]
file_path = Path(f"{folder}/{object_name}_{max_file_version}.pickle")
relationships_data_for_training_updated_model=read_dataset_from_file(file_path)
dicts.append(relationships_data_for_training_updated_model)    

"""
while current_file_version<=max_file_version:
    file_path = Path(f"{folder}/{object_name}_{current_file_version}.pickle")
    if file_path.exists():
        relationships_data_for_training_updated_model=read_dataset_from_file(file_path)
        dicts.append(relationships_data_for_training_updated_model)
    current_file_version +=1
"""

model_predictions={}
for dict in dicts:
    for k, v in dict.items():
        if k not in model_predictions.keys():
            model_predictions[k]=[]
        model_predictions[k].extend([int(x) for x in v['predictions']])

# get the model inputs in a separate object from the supervised model predictions...
relationships_data_for_training_updated_model = {
    k: pd.concat([d[k]['modelInput'] for d in dicts if k in d]).loc[lambda df: ~df.index.duplicated(keep='first')].fillna(0.0)
    for k in set().union(*dicts)
}

# now get human generated ground truth labels for the model inputs, and generate
# performance metrics by comparing the human ground truth labels with the 
# supervised model predictions
reports={}
ground_truth={}
tuned_models={}
for k in relationships_data_for_training_updated_model.keys():
    if len(relationships_data_for_training_updated_model[k])==len(model_predictions[k]):
        print(k)
        relationships_data_for_training_updated_model[k]['Flagged']=model_predictions[k]
        metrics=measure_prediction_accuracy(relationships_data_for_training_updated_model[k])
        reports[k]=metrics['report']
        ground_truth[k]=metrics['ground_truth']
        tuned_models[k]=retrain_model(k, relationships_data_for_training_updated_model)

# if necessary, fine tune a model for a particular item type
dtc=retrain_model('Variable', relationships_data_for_training_updated_model)
relationships_data_for_training_updated_model['Variable']['Flagged'] = dtc.predict(
    relationships_data_for_training_updated_model['Variable']
    )
metrics=measure_prediction_accuracy(relationships_data_for_training_updated_model[k])
        
metrics=measure_prediction_accuracy(relationships_data_for_training_updated_model[k])
record_model(model, report)

# this is code for visualising a decision tree. it could be used to compare the
# original supervised model with the fine-tuned model trained on an updated
# dataset including inputs encountered in production (i.e. post-deployment)
plt.figure(figsize=(20, 10))
tree.plot_tree(
    dtc,
    feature_names=dtc.feature_names_in_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()
