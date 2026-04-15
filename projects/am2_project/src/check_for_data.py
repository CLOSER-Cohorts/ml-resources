import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from src.logging.utility import StructuredMessage, setup_logging    
from collections import Counter;

setup_logging()
logger = logging.getLogger("am2_project")

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

def get_data():    
    with open("logs/am2_log.json") as f:
        data = [json.loads(line) for line in f]
    return data

log_entries=get_data()
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
    logger.info(StructuredMessage(description='Checking for newly available data in Colectica repository'),
        operation_type="data_check_start",
        )
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
            create_am2_input_features
        )
        from src.ml_resources.data import colectica_utility

        colectica_client = colectica_utility.C

        # Get most recent version of model...
        folder = "./projects/am2_project/models/all_item_models"
        object_name = "all_item_models"
        file_version = get_max_file_version(Path(f"{folder}"), object_name)
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        all_item_models=read_dataset_from_file(file_path)

        # Your core logic here
        #logging.info(StructuredMessage(description="Creating input features..."))
        with open("./projects/am2_project/config/am2_config.json") as f:
            project_config = json.load(f)
        # Check for new data...    
        results = check_for_newly_available_data(project_config)
        if len(results['new_item_urns'])>0:
            items = [create_item_object(x) for x in results['new_item_urns']][0:1000]
            item_types = sorted(set([colectica_client.item_code_inv(item['ItemType']) 
                for item in items]))
            all_new_am2_relationships_data={}
            for item_type in item_types:
                items_of_a_type = [x for x in items 
                    if x['ItemType']==colectica_client.item_code(item_type)]
                start_time_for_input_feature=datetime.now()
                # Create new input features...
                logger.info(StructuredMessage(message=f"Creating input features...",
                    operation_type="input_feature_creation_start",
                    status="Pending"))
                new_am2_relationships_data_single_type=create_am2_input_features(items_of_a_type, colectica_client)
                duration_input_creation=datetime.now()-start_time_for_input_feature
                logger.info(StructuredMessage(description=f"Time for input creation for {len(items_of_a_type)} items of type {item_type}",
                    operation_type=f"input feature creation_end",
                    item_type=item_type,
                    number_of_records=len(items_of_a_type),
                    status="Success",
                    duration=duration_input_creation.seconds))
                #if item_type in all_item_models.keys():
                if item_type=='Data Collection':
                    order=list(all_item_models['Category_classifier_for_error_detection']['model'].feature_names_in_)
                    new_am2_relationships_data_single_type=new_am2_relationships_data_single_type.reindex(
                        columns=order).replace({np.nan: 0}) 
                    # Make predictions on new data...
                    start_time_for_model_predictions=datetime.now()
                    logger.info(StructuredMessage(message=f"Making predictions on data...",
                        operation_type="predictions_start",
                        status="Pending"))
                    predictions=all_item_models['Category_classifier_for_error_detection']['model'].predict(
                        new_am2_relationships_data_single_type
                        )
                    duration_model_prediction=datetime.now()-start_time_for_model_predictions
                    logger.info(StructuredMessage(description=f"Time for AM2 model predictions for {len(new_am2_relationships_data_single_type)} items of type {item_type}",
                        operation_type="predictions_start",
                        status="Success",
                        duration=duration_model_prediction.seconds))
                    indices_flagged = [i for i, x in enumerate(predictions) if x == -1]    
                    anomalies=list(new_am2_relationships_data_single_type.index[indices_flagged])
                    if len(anomalies)>0:
                        print("ANOMALIES PREDICTED")
                        print(anomalies)
                        logger.info(StructuredMessage(description=f"{len(anomalies)} anomalies detected for items of type {item_type}",
                        operation_type="anomalies_detected",
                        number_number_of_anomalies=len(anomalies),
                        item_type=item_type))
                        #send_message_to_slack(str(anomalies))
                all_new_am2_relationships_data[item_type]=new_am2_relationships_data_single_type
            # Save data we just retrieved for use in training a new model
            save_versioned_pickle_file(
                all_new_am2_relationships_data,
                'am2_relationships_data_for_future_model',
                folder='./projects/am2_project/data/pending_training_data',
                )
        duration_input_creation=datetime.now()-start_time_for_input_creation
        logger.info(StructuredMessage(description="Time for entire data check process",
            status="Success",
            operation_type="data_check_end",
            duration=duration_input_creation.seconds))
        all_sweeps=get_all_sweeps()
        sweeps_not_in_project=check_for_new_sweeps(project_config, all_sweeps)
        if len(sweeps_not_in_project)>0:
            print(f"""
                The following sweeps are present in the repository, but are not included in the project:
                {sweeps_not_in_project} 
                """)
            #send_message_to_slack(f"""
            #    The following sweeps are present in the repository, but are not included in the project:
            #    {sweeps_not_in_project} 
            #    """)

        # Example task
        print(f"Hello at {datetime.now()}")

        #logging.info("Script completed successfully")
        # The command below works, but just to minimise noise on the channel, we 
        # will comment it out for now.
        #send_message_to_slack("This is a test message from code that polls the Colectica repo.")

    except Exception as e:
        #logger.exception("Script failed")
        #time_for_new_sweeps_check=datetime.now()-start_time_for_new_sweep_detection
        logger.info(StructuredMessage(description=f"Script failed: {e}",
            status="Failed"))
        #send_message_to_slack("Check for new items failed.")
        print(e)
        #send_message_to_slack(str(e))
        sys.exit(1)  # important for scheduler to detect failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="default_value")
    args = parser.parse_args()

    #setup_logging()
    main(args)