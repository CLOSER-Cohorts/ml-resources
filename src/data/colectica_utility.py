"""A set of functions to interact with the Colectica API. This is not intended to be a full client, 
but rather a set of helper functions required to interact with the API when working on
machine learning projects."""

from colectica_api import ColecticaObject
from examples.lib.utility import update_repository
USERNAME = "USERNAME"
PASSWORD = "PASSWORD"
HOSTNAME = "HOSTNAME"
C = ColecticaObject(HOSTNAME, USERNAME, PASSWORD, verify_ssl=False)

def getItemText(item_type, text_field, search_set=[], items_text={}):
    study_items = C.search_items(item_type, 
        SearchSets=search_set,
        SearchLatestVersion=True)['Results']
    for item in study_items:
        if item['AgencyId'] not in items_text.keys():
            items_text[item['AgencyId']] = {}
        if 'en-GB' in item[text_field].keys():
            items_text[item['AgencyId']][item['Identifier']] = item[text_field]['en-GB']
        elif item[text_field]!={} and len(item[text_field].keys())==0:
            items_text[item['AgencyId']][item['Identifier']] = item[text_field]

def getQuestionsForStudies(studies, all_question_summaries):
    for study in studies:
        study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]
        print(f"Getting question summaries for {study['AgencyId']}...")
        getItemText(C.item_code('Question'),
                'Summary',
                search_set = study_search_set,
                items_text = all_question_summaries
                )
    
def getVariablesForStudies(studies, all_variable_labels):
    for study in studies:
        study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]    
        print(f"Getting variable labels for {study['AgencyId']}...")
        getItemText(C.item_code('Variable'),
            'Label',
            search_set = study_search_set,
            items_text = all_variable_labels)

all_studies = C.search_items(C.item_code('Series'), SearchLatestVersion=True)['Results']
all_question_summaries={}
all_variable_labels={}
getQuestionsForStudies(all_studies, all_question_summaries)
getVariablesForStudies(all_studies, all_variable_labels)

