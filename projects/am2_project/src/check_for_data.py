import argparse
import logging
import sys
import json
from datetime import datetime
from .data.utility import (
        check_for_newly_available_data)
from src.ml_resources import (
    save_versioned_pickle_file
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
        """
        folder = "./projects/am2_project/models/all_item_models"
        object_name = "all_item_models"
        file_version = get_max_file_version(Path(f"{folder}"), object_name)
        file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
        model_package=read_dataset_from_file(file_path)
"""

        # Your core logic here
        logging.info(f"Running task with param: {args.param}")
        with open("./projects/am2_project/config/am2_config.json") as f:
            project_config = json.load(f)
        results = check_for_newly_available_data(project_config)
        if len(results['new_item_urns'])>0:
            # I need to run create_am2_input_features here : I need to 
            # have the data types as well as the urns
            items = [create_item_object(x) for x in results['new_item_urns']][0:1000]
            item_types = set([item['ItemType'] for item in items])
            new_am2_relationships_data={}
            for item_type in item_types:
                items_of_a_type = [x for x in items if x['ItemType']==item_type]
                new_am2_relationships_data=create_am2_input_features(items_of_a_type, colectica_client)
                new_relationships_data[item_type]=new_am2_relationships_data
            save_versioned_pickle_file(
                new_am2_relationships_data,
                'am2_relationships_data_for_future_model',
                folder='./projects/am2_project/data/pending_training_data',
                )
            # need to properly import the model below in future issue    
            #model.predict(results['all_am2_relationships_data'])
            
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