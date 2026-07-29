"""A set of functions to interact with the Colectica API. This is not intended to be a full client, 
but rather a set of helper functions required to interact with the API when working on
machine learning projects."""

from colectica_api import ColecticaObject
import os
import sys
import re
import json
from natsort import natsorted
from datetime import datetime
import logging
from src.logging.utility import StructuredMessage

logger = logging.getLogger("am2_project")

REQUIRED_VARS = ["COLECTICA_USERNAME", "COLECTICA_PASSWORD", "COLECTICA_HOSTNAME"]

with open("./config/secrets.json") as f:
    secrets = json.load(f)

missing = [var for var in REQUIRED_VARS if var not in secrets.keys()]

if missing:
    print("Error: Missing required config parameters:")
    for var in missing:
        print(f"  - {var}")
    sys.exit(1)

USERNAME = secrets["COLECTICA_USERNAME"]
PASSWORD = secrets["COLECTICA_PASSWORD"]
HOSTNAME = secrets["COLECTICA_HOSTNAME"]
C = ColecticaObject(HOSTNAME, USERNAME, PASSWORD, verify_ssl=False)


def get_item_text(item_type, text_field, search_set=[], items_text={}, study_items=None):
    if study_items==None:
        study_items = C.search_items(item_type,
            SearchSets=search_set,
            SearchLatestVersion=True)['Results']
    for item in study_items:
        if item['AgencyId'] not in items_text.keys():
            items_text[item['AgencyId']] = {}
        if item['Identifier'] not in items_text[item['AgencyId']].keys():
            items_text[item['AgencyId']][item['Identifier']]={}
            items_text[item['AgencyId']][item['Identifier']]['ItemType'] = item_type
            items_text[item['AgencyId']][item['Identifier']]['AgencyId'] = item['AgencyId']
            if 'en-GB' in item[text_field].keys():
                items_text[item['AgencyId']][item['Identifier']]['TextLabel'] = item[text_field]['en-GB']
            elif item[text_field]!={} and len(item[text_field].keys())==0:
                items_text[item['AgencyId']][item['Identifier']]['TextLabel'] = item[text_field]

def get_item_text_with_sweeps(item_type, text_field, search_set=[], items_text={}):
    count=0
    for search_item in search_set:
        study_items = C.search_items(item_type,
            SearchSets=search_item,
            SearchLatestVersion=True)['Results']
        for item in study_items:
            print(count)
            count=count+1
            if item['AgencyId'] not in items_text.keys():
                items_text[item['AgencyId']] = {}
            if item['Identifier'] not in items_text[item['AgencyId']].keys():
                items_text[item['AgencyId']][item['Identifier']]={}
                items_text[item['AgencyId']][item['Identifier']]['ItemType'] = item_type
                items_text[item['AgencyId']][item['Identifier']]['ContainedIn'] = search_item['identifier']
                items_text[item['AgencyId']][item['Identifier']]['AgencyId'] = item['AgencyId']
                if 'en-GB' in item[text_field].keys():
                    items_text[item['AgencyId']][item['Identifier']]['TextLabel'] = item[text_field]['en-GB']
                elif item[text_field]!={} and len(item[text_field].keys())==0:
                    items_text[item['AgencyId']][item['Identifier']]['TextLabel'] = item[text_field]

def get_items_in_containing_items(containing_items,
    all_items_text,
    text_field,
    item_type):
    """Items can be group under a number of 'containing' items, e.g. studies, sweeps,
    instruments, etc. This function retrieves labels/summaries for items contained in 
    a list of items."""
    if not isinstance(containing_items, list):
            containing_items = [containing_items]
    print("Getting labels/summaries for: ")
    print(containing_items)
    # all_question_summaries will be updated in place with the values of question summaries...
    get_item_text_with_sweeps(item_type,
                text_field,
                search_set = containing_items,
                items_text = all_items_text
                )

def get_categories_for_items(study_agency_id, item_identifiers, all_items={}, verbose=False):
    if study_agency_id not in all_items:
        all_items[study_agency_id]={}
    for index, item_identifier in enumerate(item_identifiers):
        if verbose:
            print(f"{index} of {len(item_identifiers)} items")
        code_lists=C.search_relationship_bysubject(study_agency_id,
            item_identifier,
            item_types=[C.item_code('Code Set')])
        categories_text = []
        for code_list in code_lists:
            categories=C.search_relationship_bysubject(code_list['Item1']['Item3'], code_list['Item1']['Item1'],
                  Version=code_list['Item1']['Item2'], item_types=[C.item_code('Category')])
            for category in categories:
                  category_item=C.get_item_json(category['Item1']['Item3'], category['Item1']['Item1'],
                     version=category['Item1']['Item2'])
                  if category_item['Label'] != {}:
                      categories_text.append(category_item['Label']['en-GB'])
        if item_identifier not in all_items[study_agency_id]:
            all_items[study_agency_id][item_identifier]={}
        all_items[study_agency_id][item_identifier]["ItemCategories"]=categories_text
    
def get_variables_for_studies(studies, all_variable_labels):
    for study in studies:
        study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]
        print(f"Getting variable labels for {study['AgencyId']}...")
        get_item_text(C.item_code('Variable'),
            'Label',
            search_set = study_search_set,
            items_text = all_variable_labels)

def get_topics(am1_data):
    # We get the topics for questions in the usoc study...
    #for study in project_config['Studies']:
    #    print(f"Getting topics for items in {study['AgencyId']}...")
    #    if study['AgencyId'] in am1_data.keys():
    #        colectica_utility.get_topics_for_items(list(am1_data[study['AgencyId']].keys()), study['AgencyId'], [C.item_code('Variable Group'), C.item_code('Question Group')], C, topics=am1_data, verbose=True)
    for agency_id in am1_data.keys():
        get_topics_for_items(list(am1_data[agency_id].keys()),
            agency_id,
            [C.item_code('Variable Group'), C.item_code('Question Group')], 
            C,
            topics=am1_data,
            verbose=True)
    # We get the categories for questions in our dataset (from all studies)...
    for study_agency_id in am1_data.keys():
        print(f"Getting categories for items in {study_agency_id}...")
        get_categories_for_items(study_agency_id, list(am1_data[study_agency_id].keys()), all_items=am1_data, verbose=True)
    return am1_data

def get_topics_for_items(item_identifiers,
    study_agency_id, topic_types, C, verbose=False, topics={}):
    if not isinstance(topic_types, list):
            topic_types = [topic_types]
    for index, identifier in enumerate(item_identifiers):
        if verbose:
            print(f"{index} of {len(item_identifiers)}")
        if study_agency_id not in topics.keys():
            topics[study_agency_id] = {}
        print(identifier)
        if identifier not in topics[study_agency_id].keys():
            topics[study_agency_id][identifier]={}
        topicItem=C.search_relationship_byobject(study_agency_id,
                    identifier,
                    item_types=topic_types,
                    Descriptions=True)
        topic = ""
        if len(topicItem)==1:
            print(topicItem)
            if 'en-GB' in topicItem[0]['ItemName'].keys():
                topic=topicItem[0]['ItemName']['en-GB']
            elif topicItem[0]['ItemName']!={} and len(topicItem[0]['ItemName'].keys())==0:
                topic=topicItem[0]['ItemName']
            topics[study_agency_id][identifier]['Topic'] = topic

def get_all_sweeps(batch_run_id):
    start_time_for_all_sweeps_retrieval = datetime.now()
    logger.info(StructuredMessage(message=f"Get all sweeps...",
        operation_type="get_all_sweeps_start",
        status="Pending",
        batch_run_id=batch_run_id))
    sweep_info={}
    # Get all studies...
    study_items = C.search_items([C.item_code('Series')],
            ReturnIdentifiersOnly=True,
            SearchLatestVersion=True)['Results']
    for study in study_items:
                sweep_info[study['AgencyId']]={}
                print(f"Getting sweeps for {study['AgencyId']}...")
                searchSets =[{
                    "agencyId": study['AgencyId'],
                    "identifier": study['Identifier'],
                    "version": study['Version']}]
                sweep_items = C.search_items([C.item_code('Study')],
                ReturnIdentifiersOnly=False,
                SearchSets=searchSets,
                SearchLatestVersion=True)['Results']
                sweep_names=[]
                for sweep_item in sweep_items:
                    if 'en-GB' in sweep_item['ItemName'].keys():
                        sweep_name=f"{sweep_item['ItemName']['en-GB']}"
                    else:
                        sweep_name=sweep_item['ItemName']
                    sweep_names.append(sweep_name)
                    if study['AgencyId'] not in sweep_info.keys():
                        sweep_info[study['AgencyId']]={}
                    sweep_info[study['AgencyId']][sweep_name]=sweep_item['Identifier']
    duration_of_all_sweeps_retrieval=datetime.now()-start_time_for_all_sweeps_retrieval
    logger.info(StructuredMessage(message=f"Time for getting all {len(study_items)} sweeps data",
        operation_type="get_all_sweeps_end",
        number_of_records=len(study_items),
        status="Success",
        duration=duration_of_all_sweeps_retrieval.seconds,
        batch_run_id=batch_run_id))
    return sweep_info

def get_latest_versions_of_project_sweeps(project_config, batch_run_id=None):
    # Get the latest versions of sweeps defined in the project config...
    start_time_for_latests_sweeps_retrieval = datetime.now()
    logger.info(StructuredMessage(message=f"Get latest versions of sweeps...",
        operation_type="get_latest_versions_of_project_sweeps_start",
        status="Pending",
        batch_run_id=batch_run_id))
    sweep_items = []
    for study, sweeps in project_config["ItemsForTrainingAndTest"]["Sweeps"].items():
        for sweep_name, sweep_id in sweeps.items():
            print(f"{study}, {sweep_name}, {sweep_id}")
            latest_version_of_sweep=C.get_item_json(
                study,
                sweep_id
            )
            sweep_items.append({
                 "agencyId": latest_version_of_sweep['AgencyId'],
                 "identifier": latest_version_of_sweep['Identifier'],
                 "version": latest_version_of_sweep['Version']
             })
    duration_of_new_sweeps_check=datetime.now()-start_time_for_latests_sweeps_retrieval
    logger.info(StructuredMessage(message=f"Time for getting latest versions of {len(sweep_items)} sweeps",
    operation_type="get_latest_versions_of_project_sweeps_end",
    number_of_records=len(sweep_items),
    status="Success",
    duration=duration_of_new_sweeps_check.seconds,
    batch_run_id=batch_run_id))
    return sweep_items


def obtain_items_from_colectica(batch_run_id, item_types=[], search_set_items=[]):
    start_time_for_items_retrieval = datetime.now()   
    logger.info(StructuredMessage(message=f"Obtain items from sweep...",
        operation_type="obtain_items_from_colectica_end",
        status="Pending",
        batch_run_id=batch_run_id))  
    items=[]
    item_types_for_project = [C.item_code(item_type) for item_type in item_types]
    #print(item_types_for_project)
    print(f"Searching in {search_set_items}")
    items = C.search_items(item_types_for_project,
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=search_set_items)['Results']
    duration_of_items_retrieval=datetime.now()-start_time_for_items_retrieval
    logger.info(StructuredMessage(message=f"Time for getting items of type {item_types}",
    operation_type="obtain_items_from_colectica_end",
    number_of_records=len(items),
    status="Success",
    duration=duration_of_items_retrieval.seconds,
    batch_run_id=batch_run_id))
    return items
