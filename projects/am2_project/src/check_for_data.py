import argparse
import logging
import sys
import json
from datetime import datetime
from .data.utility import (
        check_for_newly_available_data)

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
        check_for_newly_available_data(project_config)

        # Example task
        print(f"Hello at {datetime.now()}")

        logging.info("Script completed successfully")

    except Exception as e:
        logging.exception("Script failed")
        sys.exit(1)  # important for scheduler to detect failure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="default_value")
    args = parser.parse_args()

    setup_logging()
    main(args)