import argparse
import logging
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
import numpy as np
from src.logging.utility import StructuredMessage, setup_logging
from collections import Counter;
import traceback
import mlflow
from evidently import Report
from evidently.metrics import ValueDrift
import tracemalloc

logger=setup_logging()

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

def get_logs():
    with open("./logs/am2_log.json") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]
        return records

def are_series_the_same(series1, series2):
    return (
    series1.nunique(dropna=False) == 1 and
    series2.nunique(dropna=False) == 1 and
    series1.iloc[0] == series2.iloc[0]
    )


"""
log_entries=get_logs()
operation_messages=[x['message'] for x in log_entries if "operation_type" in x['message'].keys()]
message_types=set([x['operation_type'] for x in operation_messages])
for x in message_types:
    number_of_x=len([y for y in operation_messages if y['operation_type']==x])
    statuses=[y['status'] for y in operation_messages if y['operation_type']==x]
    duration=[y['duration'] for y in operation_messages if y['operation_type']==x]
    print(f"Number of {x}: {number_of_x}")
    success_rate=dict(Counter(statuses))
    print(f"Success rate: {success_rate}")
    print(f"Durations: {get_summary_stats(np.array(duration))}")
"""

def create_item_object(urn_and_item_type):
    return {
        "AgencyId": urn_and_item_type["Urn"].split(":")[2],
        "Identifier": urn_and_item_type["Urn"].split(":")[3],
        "Version": urn_and_item_type["Urn"].split(":")[4],
        "ItemType": urn_and_item_type["ItemType"]
        }

def check_for_new_sweeps(project_config, all_sweeps):
    sweep_agency_ids_in_repository=[]
    for study, sweeps in all_sweeps.items():
        for sweep_name, sweep_id in sweeps.items():
            sweep_agency_ids_in_repository.append(f"{study}:{sweep_id}:{sweep_name}")
    sweep_agency_ids_for_project=[]
    for study, sweeps in project_config["ItemsForTrainingAndTest"]["Sweeps"].items():
        for sweep_name, sweep_id in sweeps.items():
            sweep_agency_ids_for_project.append(f"{study}:{sweep_id}:{sweep_name}")
    sweeps_not_in_project=[x for x in sweep_agency_ids_in_repository if x not in sweep_agency_ids_for_project]
    return sweeps_not_in_project

def main(args):
    batch_run_id = str(uuid.uuid4())
    logger.info(StructuredMessage(message='Checking for newly available data in Colectica repository',
        operation_type="data_check_start",
        batch_run_id=batch_run_id
        ))
    start_time_for_input_creation=datetime.now()
    try:
        from src.slack.utility import send_message_to_slack
        from .data.utility import (
            check_for_newly_available_data)
        from src.ml_resources import (
            read_dataset_from_file,
            save_versioned_pickle_file,
            get_max_file_version,
            get_all_sweeps
        )
        from projects.am2_project.src.data.utility import (
            create_am2_input_features,
            get_summary_stats
        )
        from src.ml_resources.data import colectica_utility

        print("Get mlflow client...")
        colectica_client = colectica_utility.C
        with open("./config/config.json") as f:
            general_config = json.load(f)
        mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
        mlflow_client = mlflow.MlflowClient()

        # Get most recent version of model...
        folder = "./projects/am2_project/models/all_item_models"
        object_name = "all_item_models"
        file_version = get_max_file_version(Path(f"{folder}"), object_name)
        if file_version>0:
            file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
            all_item_models=read_dataset_from_file(file_path)
        else:
            all_item_models={}

        for item_type in all_item_models.keys():
            # For now we just assume that the most recent version of the model in the
            # all_item_models folder contains the live version
            #latest_versions=mlflow_client.get_latest_versions(model_name)
            model_versions=mlflow_client.search_model_versions(f"name='{item_type}_error_detection'")
            print(f"{item_type}, {len(model_versions)}")
            if len(model_versions)>0:
                latest_version = max(model_versions, key=lambda v: int(v.version)).version
                print(f"SET ALIAS FOR {item_type}")
                mlflow_client.set_registered_model_alias(
                name=f"{item_type}_error_detection", alias="live", version=latest_version
                )
        
        records=get_logs()

        # Get anomalies already flagged in logging data
        items_already_flagged=[x['message'] for x in records if
                'operation_type' in x['message'].keys() and
                x['message']['operation_type']=="anomaly_confirmation" and 
                'item_id' in x['message'].keys()]
        #items_already_flagged_ids=[x['item_id'].split(":")[2:-1] for x in items_already_flagged]
        #print(items_already_flagged)
        with open("./projects/am2_project/config/am2_config.json") as f:
            project_config = json.load(f)
        tracemalloc.start()
        # Check for new data...    
        results = check_for_newly_available_data(project_config, batch_run_id)
        # GET RID OF [0:10] TO GET EVERYTHING
        if len(results['new_item_urns'])>0:
        #if True:
            print("NEW RESULTS")
            print(results['new_item_urns'])
            items = [create_item_object(x) for x in results['new_item_urns']]
            #items = [create_item_object(x) for x in results['all_item_urns']]
            items_agency_ids=[[x['AgencyId'], x['Identifier']] for x in items]
            flagged_anomalies_that_were_updated=[x for x in items_already_flagged if 
                x['item_id'].split(":")[2:-1] in items_agency_ids]
            """
            for updated_item in flagged_anomalies_that_were_updated:
                updated_items_dict=[{"AgencyId": x.split(":")[2],
                    "Identifier": x.split(":")[3],
                    "Version": x.split(":")[4],
                    "ItemType": colectica_client.item_code(updated_item['item_type'])} for x in updated_item['all_similar_items']
                    ]
                input_features_for_updated_items=create_am2_input_features(updated_items_dict, colectica_client, logger)
                input_columns_not_in_model = [x for x in input_features_for_updated_items.columns 
                    if x not in all_item_models['Question']['model'].feature_names_in_.columns]
                input_features_for_updated_items[input_columns_not_in_model] = 0
                print(i)
                predictions=all_item_models[updated_item['item_type']]['model'].predict(
                    input_features_for_updated_items
                )
                if set(predictions.tolist())=={1}:
                    logger.info(StructuredMessage(message=f"Found updated anomaly",
                        operation_type="anomaly potentially fixed",
                        agency_id=updated_item[0],
                        identifier=updated_item[1],
                        status="potentially_fixed_anomaly",
                        batch_run_id=batch_run_id))
            """
            # Only check items for which a) there is fresh data and b) we have trained a model
            item_types = sorted(set([colectica_client.item_code_inv(item['ItemType']) for item in items]))
            all_new_am2_relationships_data={}
            new_items_for_performance_assessment={}
            for item_type in [x for x in item_types if x in list(all_item_models.keys())]:
                items_of_a_type = [x for x in items 
                    if x['ItemType']==colectica_client.item_code(item_type)]
                start_time_for_input_feature=datetime.now()
                # Create new input features...
                logger.info(StructuredMessage(message=f"Creating input features...",
                    operation_type="input_feature_creation_start",
                    status="Pending",
                    batch_run_id=batch_run_id))
                print("ITEM TYPE")
                print(item_type)
                print(items_of_a_type)
                new_am2_relationships_data_single_type=create_am2_input_features(items_of_a_type, 
                   colectica_client,
                   logger,
                   batch_run_id)
                duration_input_creation=datetime.now()-start_time_for_input_feature
                logger.info(StructuredMessage(description=f"Time for input creation for {len(items_of_a_type)} items of type {item_type}",
                    operation_type=f"input feature creation_end",
                    item_type=item_type,
                    number_of_records=len(items_of_a_type),
                    status="Success",
                    duration=duration_input_creation.seconds,
                    batch_run_id=batch_run_id))
                if duration_input_creation.seconds>0 and len(items_of_a_type)/duration_input_creation.seconds < 3:
                    send_message_to_slack(f"Input feature creation throughput for {item_type} has fallen below 3 seconds.")
                # Perform data drift checks...
                X_reference = all_item_models[item_type]['data'].reset_index(drop=True)
                X_input = new_am2_relationships_data_single_type.reset_index(drop=True)
                metrics=[]
                for column in all_item_models[item_type]['data'].columns:
                    if column in X_input.columns and not are_series_the_same(X_reference[column], 
                        X_input[column]):
                        metrics.extend([
                            ValueDrift(column=column, method="psi"),
                            ValueDrift(column=column, method="chisquare")])
                report = Report(
                        metrics=metrics
                )
                print(item_type)
                print("X_reference")
                print(X_reference.columns)
                print("X_input")
                print(X_input.columns)
                input_columns_not_in_reference = [x for x in X_input.columns if x not in X_reference.columns]
                X_reference[input_columns_not_in_reference] = 0
                reference_columns_not_in_input = [x for x in X_reference.columns if x not in X_input.columns]
                X_input[reference_columns_not_in_input] = 0
                if len(X_reference)>0 and len(X_input)>0:
                    snapshot = report.run(
                        reference_data=X_reference.astype('category'),
                        current_data=X_input.astype('category'),
                    )
                    for x in snapshot.dict()['metrics']:
                        metric=x['config']['method']
                        drift_present_psi=False
                        drift_present_chi_square=False
                        if metric=='psi':
                            drift_present_psi=x['value'] > project_config['Thresholds'][item_type]['Psi']#x['config']['threshold']
                        elif metric=='chisquare':
                            drift_present_chi_square=x['value'] < project_config['Thresholds'][item_type]['ChiSquare'] # x['config']['threshold']
                        if drift_present_psi and drift_present_chi_square:
                            drift_alert_message=f"{x['config']['column']}, {x['metric_name']}: value of {x['value']} suggests possible data drift"
                            print(drift_alert_message)
                            send_message_to_slack(drift_alert_message)
                            logger.info(StructuredMessage(description="Data drift alert",
                                status="Warning",
                                operation_type="data_drift_check",
                                drift_alert_message=drift_alert_message,
                                batch_run_id=batch_run_id
                            ))
                #registered_models = mlflow_client.search_registered_models()
                #all_registered_model_types=[model.name.removesuffix("_error_detection") for model in registered_models]
                registered_models=all_item_models
                all_registered_model_types=all_item_models.keys()
                print("REGISTERED MODELS: ")
                print(all_registered_model_types)
                print(new_am2_relationships_data_single_type)
                if item_type in all_registered_model_types and len(new_am2_relationships_data_single_type)>0:
                    order=list(all_item_models[item_type]['model'].feature_names_in_)
                    new_am2_relationships_data_single_type=new_am2_relationships_data_single_type.reindex(
                        columns=order).replace({np.nan: 0}) 
                    # Make predictions on new data...
                    start_time_for_model_predictions=datetime.now()
                    logger.info(StructuredMessage(message=f"Making predictions on data...",
                        operation_type="predictions_start",
                        status="Pending",
                        batch_run_id=batch_run_id))
                    #model = mlflow.sklearn.load_model(
                    #    model_uri=f"models:/{item_type}_error_detection@live")
                    predictions=all_item_models[item_type]['model'].predict(
                        new_am2_relationships_data_single_type
                        )
                    duration_model_prediction=datetime.now()-start_time_for_model_predictions
                    logger.info(StructuredMessage(description=f"Time for AM2 model predictions for {len(new_am2_relationships_data_single_type)} items of type {item_type}",
                        operation_type="predictions_start",
                        status="Success",
                        duration=duration_model_prediction.seconds,
                        batch_run_id=batch_run_id))
                    indices_flagged = [i for i, x in enumerate(predictions) if x == -1]    
                    anomalies=list(new_am2_relationships_data_single_type.index[indices_flagged])
                    if len(anomalies)>0:
                        print("ANOMALIES PREDICTED")
                        print(anomalies)
                        logger.info(StructuredMessage(description=f"{len(anomalies)} anomalies detected for items of type {item_type}",
                        operation_type="anomalies_detected",
                        number_number_of_anomalies=len(anomalies),
                        item_type=item_type,
                        batch_run_id=batch_run_id))
                        send_message_to_slack(f"The following possible anomalies of type {item_type} were detected:")
                        send_message_to_slack(str(anomalies))
                if len(new_am2_relationships_data_single_type)>0:        
                    all_new_am2_relationships_data[item_type]=new_am2_relationships_data_single_type
                    new_items_for_performance_assessment[item_type]={
                        "modelInput": new_am2_relationships_data_single_type,
                        "predictions": predictions
                    }
            # Save data we just retrieved for use in training a new model
            save_versioned_pickle_file(
                all_new_am2_relationships_data,
                'am2_relationships_data_for_future_model',
                folder='./projects/am2_project/data/pending_training_data',
                )
            save_versioned_pickle_file(
                new_items_for_performance_assessment,
                'new_items',
                folder='./projects/am2_project/data/pending_training_data',
                )
        duration_input_creation=datetime.now()-start_time_for_input_creation
        current, peak = tracemalloc.get_traced_memory()
        logger.info(StructuredMessage(description="Time for entire data check process",
            status="Success",
            operation_type="data_check_end",
            duration=duration_input_creation.seconds,
            peak_memory_usage=peak,
            batch_run_id=batch_run_id))
        tracemalloc.reset_peak()
        all_sweeps=get_all_sweeps(batch_run_id)
        sweeps_not_in_project=check_for_new_sweeps(project_config, all_sweeps)
        if len(sweeps_not_in_project)>0:
            print(f"""
                The following sweeps are present in the repository, but are not included in the project:
                {sweeps_not_in_project} 
                """)
            send_message_to_slack(f"""
                The following sweeps are present in the repository, but are not included in the project:
                {sweeps_not_in_project} 
                """)

        logging.info("Script completed successfully")
        # The command below works, but just to minimise noise on the channel, we 
        # will comment it out for now.
        #send_message_to_slack("This is a test message from code that polls the Colectica repo.")

    except Exception as e:
        #logger.exception("Script failed")
        send_message_to_slack("Check for new items failed.")
        stack_trace = traceback.format_exc()
        print(stack_trace)
        logger.info(StructuredMessage(description=f"Script failed: {str(stack_trace)}",
            status="Failed",
            batch_run_id=batch_run_id))
        send_message_to_slack(str(stack_trace))
        sys.exit(1)  # important for scheduler to detect failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="default_value")
    args = parser.parse_args()

    #setup_logging()
    main(args)