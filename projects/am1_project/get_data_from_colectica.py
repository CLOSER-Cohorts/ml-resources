from ml_resources.data import colectica_utility
from ml_resources import save_versioned_pickle_file
colectica_client = colectica_utility.C


study_index=3
# We get all the studies...
all_studies = colectica_client.search_items(colectica_client.item_code('Series'), SearchLatestVersion=True)['Results']
am1_data={}
# We get the questions for the first study (should be usoc)
colectica_utility.get_questions_for_studies(all_studies[0], am1_data, "Summary")

# We get the topics for questions in the usoc study...

for study in [all_studies[study_index]]:
    print(f"Getting topics for items in {study['AgencyId']}...")
    colectica_utility.get_topics_for_items(list(am1_data[study['AgencyId']].keys()),
        study['AgencyId'],
        colectica_client.item_code('Question Group'), 
        colectica_client, 
        topics=am1_data,
        verbose=True)

for study_agency_id in am1_data.keys():
    print(f"Getting categories for questions in {study_agency_id}...")
    get_categories_for_questions(study_agency_id, list(am1_data[study_agency_id].keys()),
        am1_data, verbose=True)

    #colectica_utility.get_categories_for_questions(study['AgencyId'], list(am1_data[study['AgencyId']].keys())[0:10],
    #    am1_data, verbose=True)


# Assuming that we got the topics for usoc questions, here is how we would save them into a 
# versioned pickle file... 
save_versioned_pickle_file(am1_data, 'am1_data', folder='../projects/am1_project/data')

