from src.ml_resources.data import colectica_utility
from src.ml_resources import read_dataset_from_file, save_versioned_pickle_file
import mlflow
import json

colectica_client = colectica_utility.C

with open("projects/am1_project/config/am1_config.json") as f:
    project_config_am1 = json.load(f)

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


am1_data_new=colectica_utility.get_topics(am1_data)

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



with open("./config/config.json") as f:
    general_config = json.load(f)

mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
mlflow_client = mlflow.MlflowClient()
trainedModel = mlflow.sklearn.load_model(
        model_uri="models:/logistic_regression_for_topic_classification/2")


def check_for_new_am1_data():
    with open("projects/am1_project/config/am1_config.json") as f:
        project_config = json.load(f)
    mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
    mlflow_client = mlflow.MlflowClient()
    model_version = mlflow_client.get_model_version(
    name="logistic_regression_for_topic_classification",
    version="5"
    )
    training_data_file = run.data.tags.get("mlflow.note.content")
    deployed_model_data=read_dataset_from_file(training_data_file)
    am1_data={}
    colectica_utility.get_items_in_containing_items(project_config['Studies'],
        am1_data,
        "Summary",
        colectica_client.item_code('Question'))
    colectica_utility.get_items_in_containing_items(project_config['Studies'],
        am1_data,
        "Label",
        colectica_client.item_code('Variable'))
    all_colectica_ids=[]
    for subdict in am1_data.values():
        for key in subdict.keys():
            all_colectica_ids.append(key)
    all_training_data_ids=[]
    for subdict in deployed_model_data.values():
        for key in subdict.keys():
            all_training_data_ids.append(key)
    colectica_data_not_in_training_data=[x for x in all_colectica_ids if x not in all_training_data_ids]
    return(colectica_data_not_in_training_data)


# Assuming that we got the topics for usoc questions, here is how we would save them into a 
# versioned pickle file... 

save_versioned_pickle_file(am1_data_new, 'am1_data_new', folder='./projects/am1_project/data')