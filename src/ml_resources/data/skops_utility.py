from skops.io import dump
import json
import re
import pandas as pd
from pathlib import Path

def get_max_folder_version(folder_path, object_name):
    pattern = re.compile(rf"^{re.escape(object_name)}_(\d+)$")
    max_version = 0
    for item in folder_path.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                version = int(match.group(1))
                max_version = max(max_version, version)
    return max_version

def save_versioned_model_files(model_data, object_name, folder='.'):
    folder_path = Path(f"{folder}/{object_name}/")
    folder_path.mkdir(parents=True, exist_ok=True)
    # Pattern to match files like object_name_3.pickle
    max_version=get_max_folder_version(folder_path, object_name)
    new_version = max_version + 1
    folder_path = Path(f"{folder}/{object_name}/{object_name}_{new_version}")
    folder_path.mkdir(exist_ok=True)
    models = {
        item_type: model_bundle["model"]
        for item_type, model_bundle in model_data.items()
        }
    models_file = folder_path / "all_models.skops"
    dump(models, models_file)
    models_metadata = {
        item_type: model_bundle["metadata"]
        for item_type, model_bundle in model_data.items()
        }
    metadata_file = folder_path / "metadata.json"
    with open(folder_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(models_metadata, f, indent=4)
        models_training_data = {
            item_type: model_bundle["training_data"]
            for item_type, model_bundle in model_data.items() if len(model_bundle["training_data"])>0
            }
        for name, df in models_training_data.items():
            filename = folder_path / f"{name}_training.parquet"
            df.to_parquet(filename)
        models_test_data = {
            item_type: model_bundle["test_data"]
            for item_type, model_bundle in model_data.items() if len(model_bundle["test_data"])>0
            }
        #data_file = folder_path / "data.json"
        for name, df in models_test_data.items():
            filename = folder_path / f"{name}_test.parquet"
            df.to_parquet(filename)
    

    
    