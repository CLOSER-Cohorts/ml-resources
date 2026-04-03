import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from .data.utility import (
        check_for_newly_available_data)
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    get_max_file_version
    )
from src.slack.utility import send_message_to_slack
from projects.am2_project.src.data.utility import (
        create_am2_input_features
        )
from src.ml_resources.data import colectica_utility

colectica_client = colectica_utility.C

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("script.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_item_object(urn_and_item_type):
    return {
        "AgencyId": urn_and_item_type["Urn"].split(":")[2],
        "Identifier": urn_and_item_type["Urn"].split(":")[3],
        "Version": urn_and_item_type["Urn"].split(":")[4],
        "ItemType": urn_and_item_type["ItemType"]
        }

def main(args):
    logging.info("Checking for newly available data in Colectica repository")

    try:
        # Get most recent version of model...
        folder = "./projects/am2_project/models/all_item_models"
        object_name = "all_item_models"
        file_version = get_max_file_version(Path(f"{folder}"), object_name)
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        all_item_models=read_dataset_from_file(file_path)

        # Your core logic here
        logging.info(f"Running task with param: {args.param}")
        with open("./projects/am2_project/config/am2_config.json") as f:
            project_config = json.load(f)
        results = check_for_newly_available_data(project_config)
        if len(results['new_item_urns'])>0:
            items = [create_item_object(x) for x in results['new_item_urns']][0:500]
            item_types = sorted(set([colectica_client.item_code_inv(item['ItemType']) 
                for item in items]))
            all_new_am2_relationships_data={}
            for item_type in item_types:
                items_of_a_type = [x for x in items 
                    if x['ItemType']==colectica_client.item_code(item_type)]
                new_am2_relationships_data_single_type=create_am2_input_features(items_of_a_type, colectica_client)
                new_am2_relationships_data_single_type['ItemType']=0
                if item_type in all_item_models.keys():
                    order=list(all_item_models[item_type]['model'].feature_names_in_)
                    new_am2_relationships_data_single_type=new_am2_relationships_data_single_type.reindex(
                        columns=order).replace({np.nan: 0}) 
                    predictions=all_item_models[item_type]['model'].predict(
                        new_am2_relationships_data_single_type
                        )
                    indices_flagged = [i for i, x in enumerate(predictions) if x == -1]    
                    anomalies=list(new_am2_relationships_data_single_type.index[indices_flagged])
                    if len(anomalies)>0:
                        print("ANOMALIES PREDICTED")
                        print(anomalies)
                        #send_message_to_slack(str(anomalies))
                all_new_am2_relationships_data[item_type]=new_am2_relationships_data_single_type
            save_versioned_pickle_file(
                all_new_am2_relationships_data,
                'am2_relationships_data_for_future_model',
                folder='./projects/am2_project/data/pending_training_data',
                )
            
        # Example task
        print(f"Hello at {datetime.now()}")

        logging.info("Script completed successfully")
        # The command below works, but just to minimise noise on the channel, we 
        # will comment it out for now.
        #send_message_to_slack("This is a test message from code that polls the Colectica repo.")

    except Exception as e:
        logging.exception("Script failed")
        sys.exit(1)  # important for scheduler to detect failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="default_value")
    args = parser.parse_args()

    setup_logging()
    main(args)