from ml_resources.data import colectica_utility
from ml_resources import save_versioned_pickle_file
colectica_client = colectica_utility.C

# We get all the studies...
all_studies = colectica_client.search_items(colectica_client.item_code('Series'), SearchLatestVersion=True)['Results']
am1_data={}
# We get the questions for the first study (should be usoc)
colectica_utility.get_questions_for_studies(all_studies[0], am1_data, "Summary")
save_versioned_pickle_file(usoc_question_summaries, 'usoc_question_summaries', folder='../data')

# We get the topics for questions in the usoc study...
item_topics={}
for study in [all_studies[0]]:
    print(f"Getting topics for items in {study['AgencyId']}...")
    colectica_utility.get_topics_for_items(list(am1_data[study['AgencyId']].keys()),
        study['AgencyId'],
        colectica_client.item_code('Question Group'), 
        colectica_client, 
        topics=am1_data,
        verbose=True)

# Assuming that we got the topics for usoc questions, here is how we would save them into a 
# versioned pickle file... 
save_versioned_pickle_file(item_topics, 'am1_data', folder='../data')

