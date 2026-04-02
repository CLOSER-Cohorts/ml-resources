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

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("script.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main(args):
    logging.info("Checking for newly available data in Colectica repository")

    try:
        # Your core logic here
        logging.info(f"Running task with param: {args.param}")
        with open("./projects/am2_project/config/am2_config.json") as f:
            project_config = json.load(f)
        results = check_for_newly_available_data(project_config)
        if len(results['new_item_urns'])>0:
            save_versioned_pickle_file(
                results['all_am2_relationships_data'],
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