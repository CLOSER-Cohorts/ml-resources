"""A set of functions to interact with the Colectica API. This is not intended to be a full client, 
but rather a set of helper functions required to interact with the API when working on
machine learning projects."""

from colectica_api import ColecticaObject
import os
import sys
import re
from natsort import natsorted

REQUIRED_VARS = ["COLECTICA_USERNAME", "COLECTICA_PASSWORD", "COLECTICA_HOSTNAME"]

missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]

if missing:
    print("Error: Missing required environment variables:")
    for var in missing:
        print(f"  - {var}")
    sys.exit(1)

USERNAME = os.environ.get("COLECTICA_USERNAME")
PASSWORD = os.environ.get("COLECTICA_PASSWORD")
HOSTNAME = os.environ.get("COLECTICA_HOSTNAME")
C = ColecticaObject(HOSTNAME, USERNAME, PASSWORD, verify_ssl=False)

def get_item_text(item_type, text_field, search_set=[], items_text={}):
    study_items = C.search_items(item_type, 
        SearchSets=search_set,
        SearchLatestVersion=True)['Results']
    for item in study_items:
        if item['AgencyId'] not in items_text.keys():
            items_text[item['AgencyId']] = {}
        if item['Identifier'] not in items_text[item['AgencyId']].keys():
            items_text[item['AgencyId']][item['Identifier']]={}
            if 'en-GB' in item[text_field].keys():
                items_text[item['AgencyId']][item['Identifier']][text_field] = item[text_field]['en-GB']
            elif item[text_field]!={} and len(item[text_field].keys())==0:
                items_text[item['AgencyId']][item['Identifier']][text_field] = item[text_field]

def get_questions_in_containing_items(containing_items, all_question_summaries, text_field):
    """Questions can be group under a number of 'containing' items, e.g. studies, sweeps,
    instruments, etc. This function retrieves summaries for questions contained in a list of items."""
    if not isinstance(containing_items, list):
            containing_items = [containing_items]
    print(f"Getting question summaries for {[f"Agency: {item['AgencyId']}, Identifier: {item['Identifier']}, Version: {item['Version']}" for item in containing_items]}...")
    # all_question_summaries will be updated in place with the values of question summaries...
    get_item_text(C.item_code('Question'),
                text_field,
                search_set = containing_items,
                items_text = all_question_summaries
                )

def get_categories_for_questions(study_agency_id, question_identifiers, all_items={}, verbose=False):
    if study_agency_id not in all_items:
        all_items[study_agency_id]={}
    for index, question_identifier in enumerate(question_identifiers):
        if verbose:
            print(f"{index} of {len(question_identifiers)} questions")
        code_lists=C.search_relationship_bysubject(study_agency_id,
            question_identifier,
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
        if question_identifier not in all_items[study_agency_id]:
            all_items[study_agency_id][question_identifier]={}
        all_items[study_agency_id][question_identifier]["QuestionCategories"]=categories_text
    
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

def get_topics_for_items(item_identifiers, study_agency_id, topic_type, C, verbose=False, topics={}):
    for index, identifier in enumerate(item_identifiers):
        if verbose:
            print(f"{index} of {len(item_identifiers)}")
        if study_agency_id not in topics.keys():
            topics[study_agency_id] = {}
        if identifier not in topics[study_agency_id].keys():
            topics[study_agency_id][identifier]={}
        topicItem=C.search_relationship_byobject(study_agency_id,
                    identifier,
                    item_types=[topic_type],
                    Descriptions=True)
        topic = ""
        if len(topicItem)==1:
            print(topicItem)
            if 'en-GB' in topicItem[0]['ItemName'].keys():
                topic=topicItem[0]['ItemName']['en-GB']
            elif topicItem[0]['ItemName']!={} and len(topicItem[0]['ItemName'].keys())==0:
                topic=topicItem[0]['ItemName']
            topics[study_agency_id][identifier]['Topic'] = topic

def get_sweeps():
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
                        sweep_names.append(f"{sweep_item['ItemName']['en-GB']}")
                    else:
                        sweep_names.append(sweep_item['ItemName'])
                print(sweep_names)
                sweep_info[study['AgencyId']]['SweepNames']=natsorted(sweep_names)
                sweep_info[study['AgencyId']]['SweepItems']=sweep_items
    return sweep_info

def obtain_items_from_colectica(item_type):
    items= C.search_items([C.item_code(item_type)],
            ReturnIdentifiersOnly=True,
            MaxResults=1000,
            SearchLatestVersion=True)['Results']
    return items
