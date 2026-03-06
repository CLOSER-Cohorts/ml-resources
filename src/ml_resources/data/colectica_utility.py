"""A set of functions to interact with the Colectica API. This is not intended to be a full client, 
but rather a set of helper functions required to interact with the API when working on
machine learning projects."""

from colectica_api import ColecticaObject
import os
import sys

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
            if 'en-GB' in item[text_field].keys():
                items_text[item['AgencyId']][item['Identifier']] = item[text_field]['en-GB']
            elif item[text_field]!={} and len(item[text_field].keys())==0:
                items_text[item['AgencyId']][item['Identifier']] = item[text_field]

def get_questions_for_studies(studies, all_question_summaries):
    if not isinstance(studies, list):
            studies = [studies]
    for study in studies:
        study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]
        print(f"Getting question summaries for {study['AgencyId']}...")
        get_item_text(C.item_code('Question'),
                'Summary',
                search_set = study_search_set,
                items_text = all_question_summaries
                )

def get_categories_for_questions(study_agency_id, question_identifiers, all_question_categories={}):
    if study_agency_id not in all_question_categories:
        all_question_categories[study_agency_id]={}
    for index, question_identifier in enumerate(question_identifiers):
        print(index)
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
        all_question_categories[study_agency_id][question_identifier]=categories_text
    return all_question_categories

    
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

def get_topics_for_items(items, C, topics={}):
    for index, item in enumerate(items):
        print(f"{index} of {len(items)}")
        if item['ItemType']==C.item_code('Question'):
            topic_type=C.item_code('Question Group')
        else:
            topic_type=C.item_code('Variable Group')
        if item['AgencyId'] not in topics.keys():
            topics[item['AgencyId']] = {}
        if item['Identifier'] not in topics[item['AgencyId']].keys():
            topicItem=C.search_relationship_byobject(item['AgencyId'],
                    item['Identifier'],
                    item_types=[topic_type],
                    Descriptions=True)
            topic = ""
            if len(topicItem)==1:
                if 'en-GB' in item['ItemName'].keys():
                    topic=topicItem[0]['ItemName']['en-GB']
                elif item['ItemName']!={} and len(item['ItemName'].keys())==0:
                    topic=topicItem[0]['ItemName']
            topics[item['AgencyId']][item['Identifier']] = topic