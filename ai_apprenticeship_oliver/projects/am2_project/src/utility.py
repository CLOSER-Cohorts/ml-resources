from src.ml_resources.data import colectica_utility
import json
import pandas as pd
from src.ml_resources import (
    read_dataset_from_file,
    get_max_file_version)
from pathlib import Path

colectica_client = colectica_utility.C

def get_items_with_no_labels(colectica_client):
    items_with_no_labels = []
    all_questions_variables = colectica_client.search_items(
            [colectica_client.item_code('Question'), colectica_client.item_code('Variable')],
            ReturnIdentifiersOnly=True,
            SearchLatestVersion=True,
            MaxResults=0)['Results']
    count = 0        
    for item in all_questions_variables:
        count = count + 1
        group_type=[]
        if item['ItemType']==colectica_client.item_code('Question'):
            group_type=colectica_client.item_code('Question Group')
        elif item['ItemType']==colectica_client.item_code('Variable'):
            group_type=colectica_client.item_code('Variable Group')
        topic=colectica_client.search_relationship_byobject(item['AgencyId'],
                item['Identifier'],
                item_types=group_type,
                Version=item['Version'],
                Descriptions=False)
        if len(topic)==0:
            items_with_no_labels.append(item)
        print(f"Found {len(items_with_no_labels)} items with no topics in {count} items")
    return items_with_no_labels

def get_mislabelled_items(predicted_item_labels, actual_labelled_items):
    potential_mislabels=[]
    for study in predicted_item_labels.keys():
        if study in actual_labelled_items.keys():
            for identifier in predicted_item_labels[study].keys():
                if identifier in actual_labelled_items[study].keys():
                    #print(actual_labelled_items[study][identifier])
                    if ('Topic' in actual_labelled_items[study][identifier].keys() and
                            actual_labelled_items[study][identifier]['Topic'] != 
                            predicted_item_labels[study][identifier]['Topic']):
                        potential_mislabels.append({"Study": study,
                            "Identifier": identifier,
                            "PredictedTopic": predicted_item_labels[study][identifier]['Topic'],
                            "ActualTopic": actual_labelled_items[study][identifier]['Topic']})
    return potential_mislabels

def get_training_data():
    with open("./projects/am2_project/config/am2_config.json") as f:
       project_config = json.load(f)
    folder = "./projects/am2_project/data/pending_training_data/am2_relationships_data_for_future_model"
    object_name = "am2_relationships_data_for_future_model"
    current_file_version=1
    max_file_version = get_max_file_version(Path(f"{folder}"), object_name)
    dicts=[]
    while current_file_version<=max_file_version:
        file_path = Path(f"{folder}/{object_name}_{current_file_version}.pickle")
        if file_path.exists():
            relationships_data_for_training_updated_model=read_dataset_from_file(file_path)
            dicts.append(relationships_data_for_training_updated_model)
        current_file_version +=1
    relationships_data_for_training_updated_model = {
        k: pd.concat([d[k] for d in dicts if k in d]).loc[lambda df: ~df.index.duplicated(keep='first')].fillna(0.0)
        for k in set().union(*dicts)
    }
    return relationships_data_for_training_updated_model



