from src.ml_resources.data import colectica_utility
import json

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
