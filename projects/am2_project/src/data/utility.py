import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import re
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import json
import time
import logging
import random
from src.logging.utility import StructuredMessage, setup_logging
from src.ml_resources import (
    obtain_correctly_labelled_data,
    create_model_package,
    read_dataset_from_file,
    save_versioned_pickle_file,
    get_max_file_version,
    get_latest_versions_of_project_sweeps,
    obtain_items_from_colectica,
    get_all_sweeps )

#logger=setup_logging()

with open("./config/config.json") as f:
    general_config = json.load(f)

def is_float_string(value):
    float_pattern = re.compile(r"""
    ^[+-]?              # optional sign
    (
        \.\d+           # .23
        |
        0\.\d+           # 0.23
    )
    $
""", re.VERBOSE)
    return bool(float_pattern.match(value))

def create_am2_input_features(items, colectica_client, logger):
    df_relationships = pd.DataFrame()
    # Using 'enumerate(items)' to create an index may be slow due to the complexity
    # of the item objects
    count = 0
    api_latencies=[]
    if len(items)>1:
        for item in items:
            print(f"{colectica_client.item_code_inv(item['ItemType'])}")
            if item['Identifier'] != '4f1fa78e-ff60-4a85-bd3c-aace9da5955f':
                count=count+1
                print(f"{count} of {len(items)}")
                start = time.perf_counter()
                child=colectica_client.search_relationship_bysubject(item['AgencyId'], 
                    item['Identifier'],
                    Version=item['Version'])
                parent=colectica_client.search_relationship_byobject(item['AgencyId'],
                    item['Identifier'],
                    Version=item['Version'])
                print("Get descendants...")
                descendants=colectica_client.query_set(item['AgencyId'],
                    item['Identifier'],
                    version=item['Version'])
                print("Get ancestors...")
                ancestors=colectica_client.query_set(item['AgencyId'],
                    item['Identifier'],
                    version=item['Version'],
                    reverseTraversal=True)
                api_latencies.append(time.perf_counter() - start)
                descendantTypes=set([colectica_client.item_code_inv(x['Item2']) for x in descendants])
                ancestorTypes=set([colectica_client.item_code_inv(x['Item2']) for x in ancestors])
                newRow={}
                if len(parent)>0:
                    for x in list(descendantTypes) + list(ancestorTypes):
                        if x in [colectica_client.item_code_inv(y['Item2']) for y in parent+child]:
                            newRow[x] = 5.0
                        else:
                            newRow[x] = 1.0
                # Ensure all columns in new_row exist in df
                for key in newRow:
                    if key not in df_relationships.columns:
                        df_relationships[key] = 0 
                #df_relationships.loc[len(df_relationships)] = newRow
                df_relationships.loc[f"urn:ddi:{item['AgencyId']}:{item['Identifier']}:{item['Version']}"] = newRow
                df_relationships = df_relationships.replace({np.nan: 0})
            summary_stats=get_summary_stats(np.array(api_latencies))
            print(summary_stats)
            logger.info(StructuredMessage(description="Latencies for API operations involved in feature creation",
                operation_type="input_feature_creation",
                duration=time.perf_counter() - start,
                feature_count=summary_stats['count'],
                feature_api_calls_latency_mean=summary_stats['mean'].item(),
                feature_api_calls_latency_median=summary_stats['median'].item(),
                feature_api_calls_latency_std=summary_stats['std'].item(),
                feature_api_calls_latency_min=summary_stats['min'].item(),
                feature_api_calls_latency_max=summary_stats['max'].item(),
                feature_api_calls_latency_25_percentile=summary_stats['percentiles'][0],
                feature_api_calls_latency_50_percentile=summary_stats['percentiles'][1],
                feature_api_calls_latency_75_percentile=summary_stats['percentiles'][2],
                feature_api_calls_latency_95_percentile=summary_stats['percentiles'][3],
                feature_api_calls_latency_99_percentile=summary_stats['percentiles'][4]
            ))
    return df_relationships

def generate_data_for_classification(item_type, 
    pca_data,
    all_data,
    clf,
    dataset_name="data",
    graphs_directory='./projects/am2_project/graphs/'):
    print(pca_data)
    target_variables=clf.predict(pca_data)
    scores = clf.decision_function(pca_data)
    unique_data_rows = all_data.drop_duplicates()
    final_dataset=pd.DataFrame({}, columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'ItemType', 'Flagged'])
    # CREATE THE PLOT WITH THE OUTLIERS
    scatter2 = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=target_variables, s=20, edgecolor="k")
    labels=["outliers", "inliers"]
    handles, labels = scatter2.legend_elements()
    plt.axis("square")
    plt.legend(handles=handles, labels=labels, title="true class")
    plt.title(f"Outlier detection for {item_type}")
    plt.show(block=False)
    plt.savefig(f"{graphs_directory}isolation_forest_outliers_{dataset_name}_{item_type}.png")
    plt.close()
    for index, target_variable in enumerate(target_variables):
            row_for_final_dataset = pca_data[index].tolist()
            row_for_final_dataset.extend([math.dist(pca_data[index], [0,0]), 
            scores[index],
            item_type, 
            target_variable])
            final_dataset.loc[len(final_dataset)] = row_for_final_dataset
    return final_dataset

def generate_pca_data(X,
    item_type,
    dataset_name="_data",
    fit_data=True,
    graphs_directory='./projects/am2_project/graphs/',
    pca_data = PCA(n_components=2)):
    unique_relationships_profile = X.drop_duplicates().fillna(0)
    if len(unique_relationships_profile)>1:
        if fit_data:
            principalComponents = pca_data.fit_transform(
                unique_relationships_profile)
        else:
            principalComponents = pca_data.transform(
                unique_relationships_profile)
        count=0
        plt.title(f"Plot for {item_type}")
        for i in principalComponents:
            plt.scatter(i[0], i[1])
            numberInCluster=len(X[(X==unique_relationships_profile.iloc[count,:]).all(axis=1)])
            plt.text(i[0], i[1], (str(count)+": "+str(numberInCluster)))
            count=count+1
        plt.show(block=False)
        plt.savefig(f"{graphs_directory}pca_{dataset_name}_{item_type}.png")
        plt.close()
        return {"pcaFittedToData": pca_data,
            "principalComponents": principalComponents}

def train_semi_supervised_model(
    all_relationships_data,
    item_types,
    dataset_name="_",
    generate_classification_report=False,
    save_model_in_package_file=True,
    only_relabel_outliers=True,
    all_models={},
    model_class=DecisionTreeClassifier):
    all_human_labelled_data=pd.DataFrame()
    all_training_reports={}
    all_test_reports={}
    notes=""
    print(item_types)
    for item_type in item_types:
        #mlflow.set_tracking_uri("http://127.0.0.1:5001")
        mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
        #model = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        model = model_class(max_depth=10, class_weight='balanced')
        if (item_type in all_relationships_data.keys() and 
                len(all_relationships_data[item_type])>3 and 
                input(f"Do you want to process the {item_type} items? ") in ['y', 'Y']):
            print(f"Creating semi-supervised model for {item_type}")
            print(f"There are {len(all_relationships_data[item_type])} items of this type")
            # need to add check below that we are entering float value
            test_size_value=""
            while not is_float_string(test_size_value):
                test_size_value=input("What proportion of the items do you want to put aside for testing? ")
            # Generate training data, and train a decision tree
            df_relationships = all_relationships_data[item_type]
            df_relationships_unique=df_relationships.drop_duplicates().fillna(0)
            print(df_relationships_unique)
            if len(df_relationships_unique)>4:
                X_train, X_test = train_test_split(
                    df_relationships_unique, # this needs to be specific to data type it's not at present
                 test_size=float(test_size_value)
                )
            else:
                X_train=df_relationships_unique
                X_test=pd.DataFrame()
            pca_output=generate_pca_data(X_train,
                item_type, 
                dataset_name=f"{dataset_name}_training")
            if pca_output is not None:
                pca_data = pca_output['principalComponents']
                fitted_pca = pca_output['pcaFittedToData']    
                clf = IsolationForest(max_samples=100, random_state=0)
                clf.fit(pca_data)
                training_dataset_isolation_forest=generate_data_for_classification(item_type,
                    pca_data,
                    X_train,
                    clf,
                    dataset_name=dataset_name)
                training_dataset_isolation_forest.index = X_train.index
                if 'ItemType' in X_train.columns:
                    X_train_copy=X_train.copy().drop('ItemType', axis=1)
                else:
                    X_train_copy=X_train.copy()
                data_for_model=X_train_copy.join(training_dataset_isolation_forest)
                model_name=f"{item_type}_error_detection"
                training_data_description=f"{len(X_train)}_{item_type}_items"
                model_training_results=obtain_correctly_labelled_data(
                    data_for_model,
                    'We are correcting the pseudo-labelled datasets created with isolation forests',
                    'Flagged',
                    model_name,
                    training_data_description,
                    df_relationships_unique,
                    df_relationships,
                    item_type=item_type,
                    target_variable_is_binary=True,
                    categories=[-1,1],
                    only_relabel_outliers=only_relabel_outliers,
                    generate_classification_report=generate_classification_report,
                    all_reports=all_training_reports
                    )
                human_labelled_training_data=model_training_results["UserLabelledData"]
                all_human_labelled_data=pd.concat([all_human_labelled_data,
                    human_labelled_training_data])
                #human_labelled_training_data["ItemType"] = human_labelled_training_data["ItemType"].astype("category").cat.codes
                #X=pd.DataFrame(human_labelled_training_data.drop(columns=['Flagged']))
                X=pd.DataFrame(human_labelled_training_data.drop(columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'Flagged']))
                y=human_labelled_training_data[['Flagged']]
                print("FITTING MODEL")
                X["ItemType"] = X["ItemType"].astype("category").cat.codes
                save_versioned_pickle_file(X, 'train_input', folder='./projects/am1_project/data')
                model.fit(X, y)
                # Generate test data, and calculate the accuracy of the decision tree
                if len(X_test)>3:
                    print("Now we will run tests on the data we set aside...")
                    input_for_test_pca=X_test.drop_duplicates().fillna(0)
                    pca_output = generate_pca_data(input_for_test_pca,
                        item_type,
                        dataset_name=f"{dataset_name}_test",
                        fit_data=False,
                        pca_data=fitted_pca)
                    test_pca_data = pca_output['principalComponents']    
                    test_dataset_isolation_forest = generate_data_for_classification(item_type,
                        test_pca_data,
                        X_test,
                        clf,
                        dataset_name="test_data")
                    test_dataset_isolation_forest.index = input_for_test_pca.index
                    #test_dataset_isolation_forest["ItemType"] = test_dataset_isolation_forest["ItemType"].astype("category").cat.codes
                    if 'ItemType' in X_test.columns:
                        X_test_copy=X_test.copy().drop('ItemType', axis=1)
                    else:
                        X_test_copy=X_test.copy()
                    test_dataset_for_model=X_test_copy.join(test_dataset_isolation_forest)
                    test_dataset_for_model["ItemType"] = test_dataset_for_model["ItemType"].astype("category").cat.codes
                    y_pred=model.predict(pd.DataFrame(
                        test_dataset_for_model.drop(
                            columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'Flagged'])))
                    test_input=pd.DataFrame(
                        test_dataset_for_model.drop(
                            columns=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'Flagged']))
                    save_versioned_pickle_file(test_input, 'test_input', folder='./projects/am1_project/data')
                    # We replace the 'Flagged' target variable in the test dataset which contains values
                    # calculated by an IsolationForest with the predictions produced by the supervised
                    # model.
                    test_dataset_isolation_forest['Flagged']=y_pred
                    print(test_dataset_for_model)
                    test_dataset_isolation_forest=X_test_copy.join(test_dataset_isolation_forest)
                    model_test_results=obtain_correctly_labelled_data(
                        test_dataset_isolation_forest,
                        'We are testing isolation forests',
                        'Flagged',
                        model_name,
                        training_data_description,
                        df_relationships_unique,
                        df_relationships,
                        item_type=item_type,
                        target_variable_is_binary=True,
                        only_relabel_outliers=False,
                        categories=[-1,1],
                        generate_classification_report=generate_classification_report,
                        all_reports=all_test_reports
                        )
                    report = model_test_results["ResultsReport"]
                    notes=input("Write any notes you want to include in metadata here, or press 'Enter' to leave the notes field empty. ")
                    # Get rid of 'Inferred schema contains integer column(s)' warning...
                    #input_example = X[:5].copy().to_numpy(dtype="float64")
                    input_example = X[:5]
                    with mlflow.start_run():
                        # Log parameters and metrics using the MLflow APIs
                        mlflow.log_param("model_class", model_class.__name__)
                        mlflow.log_params(model.get_params())
                        mlflow.log_metric("accuracy", report['accuracy'])
                        mlflow.log_metric("macro average precision", report['macro avg']['precision'])
                        mlflow.log_metric("macro average recall", report['macro avg']['recall'])
                        mlflow.log_metric("macro average f1-score", report['macro avg']['f1-score'])
                        mlflow.log_metric("macro average support", report['macro avg']['support'])
                        mlflow.log_metric("weighted average precision", report['weighted avg']['precision'])
                        mlflow.log_metric("weighted average recall", report['weighted avg']['recall'])
                        mlflow.log_metric("weighted average f1-score", report['weighted avg']['f1-score'])
                        mlflow.log_metric("weighted average support", report['weighted avg']['support'])
                        mlflow.set_tag(
                            "training_data_url",
                            "https://s3.amazonaws.com/bucket/training-data.csv"
                        )
                        mlflow.set_tag(
                            "mlflow.note.content", (notes + " https://s3.amazonaws.com/bucket/training-data.csv") 
                        )
                        # Log the sklearn model and register it
                        model_info = mlflow.sklearn.log_model(
                            sk_model=model,
                            name=model_name,
                            input_example=input_example,
                            registered_model_name=model_name,
                            serialization_format="skops"
                        )
                    all_human_labelled_data=pd.concat([all_human_labelled_data,
                        model_test_results["UserLabelledData"]])
                if save_model_in_package_file == True:
                    model_package=create_model_package(model,
                        human_labelled_training_data,
                        'Flagged', 
                        preprocessing=["PCA"],
                        notes=notes,
                        model_version=model_name,
                        training_data_version=training_data_description,
                        training_item_ids=list(df_relationships.index))
                    save_versioned_pickle_file(model_package,
                        model_name,
                        folder='./projects/am2_project/models')
                all_models[item_type]=model_package
    if save_model_in_package_file == True:
        # Move columns to the end of the dataframe...
        cols_to_move=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore', 'ItemType', 'Flagged']
        if len(all_human_labelled_data)>0:
            all_human_labelled_data=all_human_labelled_data[[c for c in all_human_labelled_data.columns if c not in cols_to_move] + cols_to_move]
            save_versioned_pickle_file(all_human_labelled_data.replace({np.nan: 0}),
                    "all_human_labelled_data",
                    folder='./projects/am2_project/data/human_labelled_data')
            save_versioned_pickle_file(all_models,
                    "all_item_models", 
                    folder='./projects/am2_project/models')
            save_versioned_pickle_file(all_relationships_data,
                    'all_am2_relationships_data',
                    folder='./projects/am2_project/data')
            save_versioned_pickle_file(all_training_reports,
                    f"classification_report_all_items_training",
                    folder=f"./projects/am2_project/experiments/all_items")
            save_versioned_pickle_file(all_test_reports,
                    f"classification_report_all_items_test",
                    folder=f"./projects/am2_project/experiments/all_items")

def create_urn(item):
    return {
        "Urn": f"urn:ddi:{item['AgencyId']}:{item['Identifier']}:{item['Version']}",
        "ItemType": item['ItemType']
    }

def get_cached_versions_of_project_sweeps():
    folder = "./projects/am2_project/data/sweep_items_cached"
    object_name = "sweep_items_cached"
    file_version = get_max_file_version(Path(f"{folder}"), object_name)
    if file_version >0:
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        cached_sweeps=read_dataset_from_file(file_path)
    else:
        cached_sweeps=[]
    return cached_sweeps

def check_for_newly_available_data(project_config):
    all_urns_in_current_dataset=[]
    all_item_urns=[]
    new_item_urns=[]
    folder=project_config["AllModelsFileLocation"]
    object_name=project_config["AllModelsObjectName"]
    file_version=get_max_file_version(Path(f"{folder}"), object_name)
    if file_version>0:
        try:
            file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
            print(f"Reading details of trained models from {file_path}")
            all_models=read_dataset_from_file(file_path)
            for item_type, model in all_models.items():
                all_urns_in_current_dataset.extend(model['metadata']["training_item_ids"])
        except Exception as e:
            raise FileNotFoundError(f"File not found error: {e}")
    cached_sweep_items=get_cached_versions_of_project_sweeps()
    sweep_items=get_latest_versions_of_project_sweeps(project_config)
    updated_sweeps=[x for x in sweep_items if x not in random.sample(cached_sweep_items,len(cached_sweep_items)-3)]
    print("UPDATED SWEEPS")
    print(updated_sweeps)
    if len(updated_sweeps)>0:
        items=obtain_items_from_colectica(project_config["ItemTypes"], updated_sweeps)
        all_item_urns=[f"urn:ddi:{item['AgencyId']}:{item['Identifier']}:{item['Version']}"
            for item in items]
        #new_item_urns=[create_urn(x) for x in items if create_urn(x)["Urn"] not in all_urns_in_current_dataset]
        new_item_urns=[create_urn(x) for x in items]
        if len(new_item_urns)>0:
            print(f"There are {len(new_item_urns)} items are available for analysis/inclusion in the data model.")
    save_versioned_pickle_file(sweep_items, 'sweep_items_cached', folder='./projects/am1_project/data')
    return {"all_item_urns": all_item_urns,
            "all_urns_in_current_dataset": all_urns_in_current_dataset,
            "new_item_urns": new_item_urns}

def data_quality_checks(model_metadata, input_data, allowed_feature_values={0.0, 1.0, 5.0}):
    # Check if there are any missing values in the data
    # Find columns with missing values
    missing_columns=[]
    columns_missing_values=[]
    columns_with_wrong_datatypes=[]
    for input_feature in model_metadata['input_features']:
        if input_feature not in input_data.columns:
            missing_columns.append(input_feature)
        elif input_data[input_feature].isna().any():
            columns_missing_values.append(input_feature)
    if len(missing_columns)>0:
        print("The following input features are not present in the input: ")
        print(missing_columns)
    if len(columns_missing_values)>0:
        print("The following columns contain missing values: ")
        print(columns_missing_values)
    input_data_types=json.loads(input_data.dtypes.astype(str).to_json())
    for index, value in json.loads(model_metadata['feature_types']).items():
        #print(f"{index}, {value}")
        if index in input_data_types.keys():
            if input_data_types[index]!=value:
                print(f"Type mismatch between input and model for feature {index}: model is {value} but input data is {input_data_types[index]}")
                columns_with_wrong_datatypes.append(index)
            # While the input features representing relations between items are floats,
            # they must have one from a fixed set of values e.g. 0.0/1.0/5.0
            # The input features in float_input_features can have any float value, for
            # example 2.35, 12.43, etc., while str_input_features must be strings
            # We're not imposing limits on these float/string input features for now,
            # but we're specifying them in case we do so at some future point.   
            float_input_features = ['x', 'y', 'DistanceFromOrigin', 'AnomalyScore']
            str_input_features = ['ItemType']
            set_input_features = [x for x in model_metadata['input_features'] if 
                x not in float_input_features and 
                x not in str_input_features]
            values_for_input_feature = set(input_data[index])-allowed
            if index in set_input_features and values_for_input_feature-allowed != set():
                print(f"The {index} input features contain the following invalid value(s): {values_for_input_feature}")

def get_summary_stats(data):
    summary = {
    "count": len(data),
    "mean": data.mean(),
    "median": np.median(data),
    "std": data.std(),
    "min": data.min(),
    "max": data.max(),
    "percentiles": np.percentile(data, [25, 50, 75, 95, 99])
    }
    return summary
