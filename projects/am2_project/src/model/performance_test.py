from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, recall_score
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import json
from mlflow_code.mlflow_utility import record_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from projects.am2_project.src.utility import get_training_data
from src.logging.utility import StructuredMessage, setup_logging
from src.ml_resources import (
    read_dataset_from_file,
    get_max_file_version,
    get_max_folder_version,
    save_versioned_model_files,
    create_model_package,
    save_versioned_pickle_file)

"""
The code in this file is for testing the performance of the am2 anomaly detection
model on live data; there are no ground truth labels. A human must provide ground
truth labels for the live data, and these ground truth labels are compared to the
trained supervised models predictions in order to calculate performance metrics.

Run the process of generating performance metrics for the anomaly detection models
in production, and retraining those models with updated datasets that include 
production data received after the model was initially trained and deployed, by
executing the following command:

retrain_model_with_production_data_added()
"""


def combine_training_and_production_data(additional_training_data,
    original_training_data_single_type,
    model_info):
    existing_training_data=model_info['data']
    existing_training_data["ItemType"] = existing_training_data["ItemType"].astype("category").cat.codes
    existing_training_data=existing_training_data.drop(columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore'])
    #training_data_with_duplicates = original_training_data_single_type.merge(existing_training_data, how="inner")
    training_data_with_duplicates = (
        original_training_data_single_type
        .reset_index()
        .merge(existing_training_data, how="inner")
        .set_index("index")
        )
    #existing_training_labels=model_info['data']['Flagged']
    #additional_training_data=relationships_data_for_training_updated_model.drop_duplicates().drop('Flagged', axis=1)
    #additional_training_labels=relationships_data_for_training_updated_model.drop_duplicates()['Flagged']
    updated_training_data=pd.concat([training_data_with_duplicates, additional_training_data])
    #updated_training_labels=pd.concat([existing_training_labels,additional_training_labels])
    #updated_training_data['Flagged'] = updated_training_labels
    #updated_training_data_unique=updated_training_data.drop_duplicates()
    return updated_training_data
    
def retrain_model(relationships_data_for_training_updated_model, model_info):
    model=model_info['model']
    param_grid = {
    'max_depth': [2, 3, 4, 5, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
    }
    training_input=relationships_data_for_training_updated_model.drop('Flagged', axis=1)
    training_labels=relationships_data_for_training_updated_model['Flagged']
    print("TRAINING INPUT")
    print(training_input)
    print(len(training_input))
    if len([x for x in training_labels.values if x==-1])<5:
        print("NOT MUCH INPUT")
        dtc = model.fit(training_input, training_labels)
    elif -1 in training_labels.values:
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
        grid.fit(training_input, training_labels)
        dtc=grid.best_estimator_
        print("Best parameters:", grid.best_params_)
        print("Best CV performance:", grid.best_score_)
    else:
        print("No anomalies in training data to detect")
        dtc = model
    return dtc


def measure_prediction_accuracy(data_with_predictions, item_type):
    reports={}
    df_unique=data_with_predictions.drop_duplicates()
    df_y_true=data_with_predictions.copy()
    count=1
    for k,v in df_unique.iterrows():
        ground_truth=""
        print(v)
        print(k)
        while ground_truth not in ['y', 'n']:
                ground_truth=input(f"{count} of {len(df_unique)} ({item_type}), is this right? y/n ")
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

def retrieve_production_data():
    folder = "./projects/am2_project/data/pending_training_data/new_items"
    object_name = "new_items"
    current_file_version = 1
    max_file_version = get_max_file_version(Path(f"{folder}"), object_name)
    dicts=[]
    file_path = Path(f"{folder}/{object_name}_{max_file_version}.pickle")
    relationships_data_for_training_updated_model=read_dataset_from_file(file_path)
    dicts.append(relationships_data_for_training_updated_model) 
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
    for k in relationships_data_for_training_updated_model.keys():
        relationships_data_for_training_updated_model[k]['Flagged']=model_predictions[k]
    return relationships_data_for_training_updated_model
  
def retrain_model_with_production_data_added(logger):
    # Read in the input features and the trained model predictions from the production 
    # environment
    relationships_data_for_training_updated_model = retrieve_production_data()
    # Read in the previously trained models and their data
    folder = "./projects/am2_project/models/all_item_models"
    object_name = "all_item_models"
    file_version = get_max_file_version(Path(f"{folder}"), object_name)
    if file_version > 0:
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        all_item_models=read_dataset_from_file(file_path)
    else:
        all_item_models={} 
    reports={}
    ground_truth={}
    tuned_models={}
    number_of_item_types = len(list(relationships_data_for_training_updated_model.keys()))
    count = 1
    for k in list(relationships_data_for_training_updated_model.keys()):
      #if k=='Variable':
        # Now get human generated ground truth labels for the model inputs, and generate
        # supervised model predictions
        # performance metrics by comparing the human ground truth labels with the 
        print(f"Item type: {k}, {count} of {number_of_item_types}")
        count = count + 1
        # You're testing the accuracy of the trained model on data post-deployment it hasn't 
        # previously encountered; this fresh data is then combined with the existing data
        # that the model has been trained on in order to train an updated version of the model
        metrics = measure_prediction_accuracy(relationships_data_for_training_updated_model[k], 
           k)
        reports[k] = metrics['report']
        ground_truth[k]=metrics['ground_truth']
        human_labelled_training_data = relationships_data_for_training_updated_model[k]
        human_labelled_training_data['Flagged'] = metrics['ground_truth']
        original_training_data = get_training_data()
        # Retrain the model using gridsearchcv on a new training dataset, which is 
        # composed of a) the models original training data and b) the new items 
        combined_training_and_production_data=combine_training_and_production_data(
            human_labelled_training_data,
            original_training_data[k],
            all_item_models[k])
        tuned_model=retrain_model(combined_training_and_production_data, all_item_models[k])
        # Save model on MLFlow
        model_info=record_model(tuned_model,
            metrics['report'],
            model_name=f"{k}_error_detection",
            input_example=combined_training_and_production_data.drop('Flagged', axis=1)[:5])
        logger.info(StructuredMessage(description=f"Performance for {k} anomaly detection",
                    operation_type=f"{k} anomaly detection performance",
                    item_type=k,
                    model_name=model_info.name,
                    model_version=model_info.registered_model_version,
                    model_uri=model_info.model_uri,
                    number_of_records=len(relationships_data_for_training_updated_model[k]),
                    report=metrics['report']
                    )
        )
        if 'test_data' in all_item_models[k].keys():
            test_data = all_item_models[k]['test_data']
        else:
            test_data = pd.DataFrame()
        # Create a new model package with the retrained model
        model_package=create_model_package(tuned_model,
            combined_training_and_production_data,
            test_data,
            'Flagged', 
            preprocessing=["PCA"],
            notes=f"Updated {k} error detection model",
            model_version=f"{k}_error_detection",
            training_data_version=f"{len(relationships_data_for_training_updated_model[k])}_{k}_items",
            training_item_ids=list(combined_training_and_production_data.index))
        tuned_models[k]=model_package
    # 5 Save the model artefacts in separate files using save_versioned_model_files
    #save_versioned_pickle_file(tuned_models, 'tuned_models', folder='./projects/am1_project/data')
    save_versioned_model_files(tuned_models,
        "all_item_models_separate",
        folder='./projects/am2_project/models')

logger=setup_logging()    
# This is the main command for running the retraining/performance metric gathering process 
retrain_model_with_production_data_added(logger)

TODO: RUN THE ABOVE COMMAND TO RETRAIN THE MODEL WITH PRODUCTION DATA, THEN ADD SOME NEW
WAVES IN A NEW BRANCH
ITS NOT FINDING THE DRIFT REFERENCE FILES???

# to get the items that were used as test datasets when initially training the datasets
relationships_data=get_training_data()
result=list(set(relationships_data['Variable'].index) - set(training_data_with_duplicates.index))
relationships_data['Variable'].loc[result].drop_duplicates() # 28, +112=140

# open the data file for a trained model
with open("./projects/am2_project/models/all_item_models_separate/all_item_models_separate_6/metadata.json") as f:
    variable_metadata = json.load(f)
variable_df2=pd.read_parquet("./projects/am2_project/models/all_item_models_separate/all_item_models_separate_5/Variable_training.parquet")


ids_for_original_variable_model=all_item_models['Variable']['metadata']['training_item_ids']
ids_for_updated_variable_model=variable_df.index
ids_for_production_data=relationships_data_for_training_updated_model['Variable'].index
result = list(set(ids_for_production_data) - set(b))

TODO: SAVE TEST DATA IN THE VERSIONED MODEL FILES. NEED TO FIGURE OUT HOW TO GET THE TEST DATA?
PLAN
Start from the current set of waves (delete the current cache file)
Measure: the model performance
the data drift 
data quality (Missing values, invalid values, duplicates, schema changes)
prediction/concept drift (% of anomalies)
system and operation metrics
create dashboards


# The rest of this file is code for miscellaneous tasks, included for possible future convenience

# fine tune a model for a particular item type
dtc=retrain_model('Variable', relationships_data_for_training_updated_model)
relationships_data_for_training_updated_model['Variable']['Flagged'] = dtc.predict(
    relationships_data_for_training_updated_model['Variable'].drop('Flagged', axis=1)
    )
metrics=measure_prediction_accuracy(relationships_data_for_training_updated_model['Variable'])

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
