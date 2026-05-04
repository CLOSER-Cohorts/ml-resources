from src.ml_resources.data import colectica_utility
from src.ml_resources import read_dataset_from_file, save_versioned_pickle_file
import json
colectica_client = colectica_utility.C

with open("projects/am1_project/config/am1_config.json") as f:
    project_config = json.load(f)

# If you're starting from scratch, create the am1_data object...
am1_data={}
# ...otherwise read it in from a pickle file.
am1_data=read_dataset_from_file('../projects/am1_project/data/am1_data/am1_data_2.pickle')

# We get the questions for the usoc study...
colectica_utility.get_items_in_containing_items(project_config['Studies'],
    am1_data,
    "Summary",
    colectica_client.item_code('Question'))

colectica_utility.get_items_in_containing_items(project_config['Studies'],
    am1_data,
    "Label",
    colectica_client.item_code('Variable'))


am1_data_new=get_topics(am1_data)

# The above example presumes you are defining the items in your data set by
# specifying a containing item, but if you wanted to just specify a list of items,
# here is how you do it...abs
items = read_dataset_from_file('./projects/am1_project/data/sensitive_10103_ethnic_items/sensitive_10103_ethnic_items_1.pickle')
question_items=[x for x in items if x['ItemType']==C.item_code('Question')]
variable_items=[x for x in items if x['ItemType']==C.item_code('Variable')]
am1_data={}
get_item_text(C.item_code("Question"), "Summary", items_text=am1_data, study_items=question_items)
get_item_text(C.item_code("Variable"), "Label", items_text=am1_data, study_items=variable_items)
am1_data_new=get_topics(am1_data)
filtered_items=filter_values_by_length(am1_data, "TextLabel", 10)
df=convert_dictionary_to_dataframe(filtered_items)
df['HasCategories']=[int(x) for x in (df['ItemCategories']!='').tolist()]
df['ItemType']=df["ItemType"].astype("category").cat.codes
transformed_embeddings_10103 = apply_pipeline(df, ['TextLabel', 'ItemCategories'])
save_versioned_pickle_file(transformed_embeddings_10103, 'transformed_embeddings_10103', folder='./projects/am1_project/data')


def get_topics(am1_data):
    # We get the topics for questions in the usoc study...
    #for study in project_config['Studies']:
    #    print(f"Getting topics for items in {study['AgencyId']}...")
    #    if study['AgencyId'] in am1_data.keys():
    #        colectica_utility.get_topics_for_items(list(am1_data[study['AgencyId']].keys()), study['AgencyId'], [colectica_client.item_code('Variable Group'), colectica_client.item_code('Question Group')], colectica_client, topics=am1_data, verbose=True)
    for agency_id in am1_data.keys():
        colectica_utility.get_topics_for_items(list(am1_data[agency_id].keys()),
            agency_id,
            [colectica_client.item_code('Variable Group'), colectica_client.item_code('Question Group')], 
            colectica_client,
            topics=am1_data,
            verbose=True)
    # We get the categories for questions in our dataset (from all studies)...
    for study_agency_id in am1_data.keys():
        print(f"Getting categories for items in {study_agency_id}...")
        colectica_utility.get_categories_for_items(study_agency_id, list(am1_data[study_agency_id].keys()), all_items=am1_data, verbose=True)
    return am1_data

# Assuming that we got the topics for usoc questions, here is how we would save them into a 
# versioned pickle file... 

save_versioned_pickle_file(am1_data_new, 'am1_data_new', folder='./projects/am1_project/data')