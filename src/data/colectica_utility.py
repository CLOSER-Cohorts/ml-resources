"""A set of functions to interact with the Colectica API. This is not intended to be a full client, 
but rather a set of helper functions required to interact with the API when working on
machine learning projects."""

from colectica_api import ColecticaObject
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

def getTopicsForItems(items, topics={}):
    for index, item in enumerate(items):
        if item['ItemType']==C.item_code('Question'):
            topic_type=C.item_code('Question Group')
        else:
            topic_type=C.item_code('Variable Group')
        #print(f"{index} of {len(items)}")
        if item['AgencyId'] not in topics.keys():
            topics[item['AgencyId']] = {}
        if item['Identifier'] not in topics.keys():
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

all_studies = C.search_items(C.item_code('Series'), SearchLatestVersion=True)['Results']
all_question_summaries={}
all_variable_labels={}
getQuestionsForStudies(all_studies, all_question_summaries)
getVariablesForStudies(all_studies, all_variable_labels)

item_topics={}
for study in all_studies:
    study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]
    print(f"Getting topics for items in {study['AgencyId']}...")
    study_question_items = C.search_items(C.item_code('Question'),
        SearchSets=study_search_set,
        SearchLatestVersion=True)['Results']
    getTopicsForItems(study_question_items, topics=item_topics)

save_versioned_pickle_file(item_topics, 'all_item_topics')

for agencyId in item_topics.keys():
    print(f"{agencyId}, {len(list(item_topics[agencyId].items()))}, {len(list(all_question_summaries[agencyId].items()))}")