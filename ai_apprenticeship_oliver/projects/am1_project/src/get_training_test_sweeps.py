import json
from src.ml_resources import (
    get_latest_versions_of_project_sweeps,
    save_versioned_pickle_file,
    obtain_items_from_colectica)
from src.ml_resources.data import colectica_utility

colectica_client = colectica_utility.C

with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)


training_sweep_items=get_latest_versions_of_project_sweeps(project_config)
"""
am1_data={}
usoc_training_sweeps=[x for x in training_sweep_items if x['agencyId']=='uk.iser.ukhls']
colectica_utility.get_items_in_containing_items(usoc_training_sweeps,
    am1_data,
    "Summary",
    colectica_client.item_code('Question'))
colectica_utility.get_items_in_containing_items(usoc_training_sweeps,
    am1_data,
    "Label",
    colectica_client.item_code('Variable'))
am1_data=colectica_utility.get_topics(am1_data)
"""

all_study_series=colectica_client.search_items(colectica_client.item_code('Series'),
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True)['Results']

test_sweeps_dict={}
for series in all_study_series:
    series_item_id=[{"AgencyId": series['AgencyId'],
            "Identifier": series['Identifier'],
            "Version": series['Version']
    }]
    print(series_item_id)
    series_training_sweeps=[x for x in training_sweep_items 
        if x['agencyId']==series['AgencyId']]
    all_series_sweeps=colectica_client.search_items(colectica_client.item_code('Study'),
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=series_item_id)['Results']
    # because some genscot are in heaf
    filtered_sweeps=[x for x in all_series_sweeps if x['AgencyId']==series['AgencyId']]
    series_study_ids=[{'agencyId': x['AgencyId'], 
        "identifier": x['Identifier'], 
        "version": x['Version']} for x in filtered_sweeps]
    test_sweeps = [x for x in series_study_ids if x not in series_training_sweeps]
    test_sweeps_dict[series['AgencyId']]=test_sweeps

"""
CHECK TITLES OF SWEEPS
agency='uk.alspac'
len([x for x in test_sweeps_dict[agency]])
len([x for x in training_sweep_items if x['agencyId']==agency])
titles=[]
for x in test_sweeps_dict[agency]:
    y=colectica_client.get_item_json(x['agencyId'], x['identifier'])
    titles.append(y["DublinCoreMetadata"]["Title"]["en-GB"])
for x in sorted(titles):
    print(x)
"""

for series in all_study_series[11:]:
    if series['AgencyId'] not in ['uk.closer', 'uk.cls.mcs']:
        print(series['AgencyId'])
        training_data={}
        test_data={}
        training_sweeps=[x for x in training_sweep_items 
            if x['agencyId'] == series['AgencyId']]
        colectica_utility.get_items_in_containing_items(training_sweeps,
            training_data,
            "Summary",
            colectica_client.item_code('Question'))
        colectica_utility.get_items_in_containing_items(training_sweeps,
            training_data,
            "Label",
            colectica_client.item_code('Variable'))
        am1_data_new=colectica_utility.get_topics(training_data)
        save_versioned_pickle_file(training_data, 
            f'training_sweeps_{series['AgencyId']}', 
            folder='./projects/am1_project/data/sweeps/training')
        test_sweeps=test_sweeps_dict[series['AgencyId']]
        colectica_utility.get_items_in_containing_items(test_sweeps,
            test_data,
            "Summary",
            colectica_client.item_code('Question'))
        colectica_utility.get_items_in_containing_items(test_sweeps,
            test_data,
            "Label",
            colectica_client.item_code('Variable'))
        am1_data_new=colectica_utility.get_topics(test_data)
        save_versioned_pickle_file(test_data,
        f'test_sweeps_{series['AgencyId']}',
        folder='./projects/am1_project/data/sweeps/test')
        

a=read_dataset_from_file('./projects/am1_project/data/sweeps/training/training_sweeps_uk.cls.bcs70/training_sweeps_uk.cls.bcs70_1.pickle')

a=read_dataset_from_file('./projects/am1_project/data/sweeps/test/test_sweeps_uk.wchads/test_sweeps_uk.wchads_1.pickle')