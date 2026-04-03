import json
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    obtain_items_from_colectica,
    #check_for_newly_available_data,
    get_sweeps,
    get_max_file_version)
from projects.am2_project.src.data.utility import (
        create_am2_input_features,
        train_semi_supervised_model)
from src.ml_resources.data import colectica_utility

colectica_client = colectica_utility.C

#item_types_string=['Action', 'Archive', 'Category', 'Category Group', 'Category Set', 'ClassificationCorrespondenceTable', 'ClassificationFamily', 'ClassificationIndex', 'ClassificationItem', 'ClassificationLevel', 'ClassificationSeries', 'Code List Group', 'Code List Set', 'Code Set', 'Concept', 'Concept Group', 'Concept Set', 'Conceptual Component', 'Conceptual Variable', 'Conceptual Variable Group', 'Conceptual Variable Set', 'Conditional', 'Control Construct Group', 'Control Construct Set', 'Data Collection', 'Data File', 'Data Layout', 'DataCollection Methodology', 'General Instruction', 'Generation Instruction', 'Individual', 'Instruction Group', 'Instrument', 'Instrument Group', 'Instrument Set', 'Interviewer Instruction', 'Interviewer Instruction Set', 'Logical Product', 'Loop', 'Managed Representation Group', 'Managed Representation Set', 'MeasurementItem', 'MeasurementConstruct', 'Metadata Package', 'NCube', 'NCube Group', 'NCube Set', 'Organization', 'Organization Group', 'Organization Set', 'OtherMaterial', 'OtherMaterialGroup', 'OtherMaterialScheme', 'Physical Data Product', 'Physical Structure', 'PhysicalStructure Set', 'Processing Event', 'Processing Event Group', 'Processing Event Set', 'Processing Instruction Group', 'Processing Instruction Scheme', 'Project', 'Quality Standard', 'Quality Statement', 'Quality Statement Group', 'Quality Statement Set', 'Question', 'Question Activity', 'Question Block', 'Question Grid', 'Question Group', 'Question Set', 'RecordLayout', 'RecordLayout Set', 'Repeat', 'Represented Variable', 'Represented Variable Group', 'Represented Variable Set', 'Reusable Missing Value', 'Sequence', 'Series', 'Statement', 'StatisticalClassification', 'Study', 'SubSeries', 'UnitType', 'UnitTypeScheme', 'UnitTypeGroup', 'Universe', 'Universe Group', 'Universe Set', 'Variable', 'Variable Group', 'Variable Set', 'Variable Statistic', 'While']
#item_types_string=['Data File', 'Data Collection', 'Variable Statistic', 'While', 'Instrument']
#START WITH ONE DATA TYPE, E.G. DATA FILE, BUILD IN THAT



all_relationships_data={}
file_path = Path('./projects/am2_project/data/all_am2_relationships_data/all_am2_relationships_data_83.pickle')
if file_path.exists():
    all_relationships_data=read_dataset_from_file(file_path)
else:
    print("File does not exist")

a=check_for_newly_available_data(project_config)

all_sweeps=get_sweeps()
total=0
for x,y in all_sweeps.items():
   total+=len(y)


sweeps_for_training_and_test={}
for study in all_sweeps.keys():
    if study not in sweeps_for_training_and_test.keys():
        sweeps_for_training_and_test[study]={}
    for sweep_name, sweep_id in all_sweeps[study].items():
        if sweep_name not in project_config['ItemsForValidation']['Sweeps'][study]:
            sweeps_for_training_and_test[study][sweep_name]=sweep_id
json_formatted_str = json.dumps(sweeps_for_training_and_test, indent=4)
print(json_formatted_str)


# Let's check if there is new data...

question_keys_from_repository_for_dataset = []

"""
for item in items:
    if colectica_client.item_code_inv[item['ItemType']] not in all_relationships_data.keys():
        print(f"Need to create a relationships profile for {item_type} items")
        df_relationships=create_am2_input_features(items, colectica_client)
        all_relationships_data[item_type]=df_relationships
    else:
        print(f"We already have data for item type {item_type}")
        df_relationships=all_relationships_data[item_type]
    # Let's check for newly available data on Colectica...
    ids_for_newly_available_data=check_for_newly_available_data(
        [x['Identifier'] for x in items],
        [x.split(":")[1] for x in df_relationships.index])
"""


# The code below could be part of the deployed application;
# it could run as a scheduled task, and every time someone goes to a dashboard
# to view details of outliers, if there were new data available
# a message would be displayed to users
for item_type in item_types_string:
    items=obtain_items_from_colectica(item_type)
    if item_type not in all_relationships_data.keys():
        print(f"Need to create a relationships profile for {item_type} items")
        df_relationships=create_am2_input_features(items, colectica_client)
        all_relationships_data[item_type]=df_relationships
    else:
        print(f"We already have data for item type {item_type}")
        df_relationships=all_relationships_data[item_type]    
    # Let's check for newly available data on Colectica...
    ids_for_newly_available_data=check_for_newly_available_data(
        [x['Identifier'] for x in items],
        [x.split(":")[1] for x in df_relationships.index])
         
df_relationships_unique=df_relationships.drop_duplicates().fillna(0)

# This is the model we will train in semi-supervised fashion...
dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
# check_is_fitted throws an error if the model isn't fitted. The below statement should throw
# an error, but after we run train_semi_supervised_model, check_is_fitted should run and return
# nothing, without throwing an exception.
check_is_fitted(dtc)
all_training_data={}
all_principal_components={}

# Get the most up to date version of the training data

with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)

folder = "./projects/am2_project/data/pending_training_data/am2_relationships_data_for_future_model"
object_name = "am2_relationships_data_for_future_model"
file_version = get_max_file_version(Path(f"{folder}"), object_name)
file_path = Path(f"{folder}/{object_name}_{file_version}.pickle")
relationships_data_for_training_updated_model=read_dataset_from_file(file_path)

train_semi_supervised_model(
    relationships_data_for_training_updated_model,
    project_config['ItemTypes'],
    dataset_name="wip",
    generate_classification_report=True,
    save_model_in_package_file=True)
# dtc was fitted in train_semi_supervised_model so the below check_is_fitted command should
# run and not throw an error.
check_is_fitted(dtc)
save_versioned_pickle_file(all_am2_relationships_data,
        'all_am2_relationships_data', folder='./projects/am2_project/data')

future_data=read_dataset_from_file('projects/am2_project/data/pending_training_data/am2_relationships_data_for_future_model/am2_relationships_data_for_future_model_3.pickle')
# The commands below are useful for reading the supervised model trained by
# train_semi_supervised_model above into memory for inspection
model_package=read_dataset_from_file(
    './projects/am2_project/models/Instrument_classifier_for_error_detection/Instrument_classifier_for_error_detection_9.pickle')
dtc=model_package['model']
plt.figure(figsize=(10, 10))
tree.plot_tree(
    dtc,
    class_names=["2", "3"],
    filled=True,
    feature_names=['x', 'y', 'ItemType', 'Distance']
)
plt.show()

def my_func(all_relationships_dataNEW):
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 0).any(axis=0)]
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 1).any(axis=0)]
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 5).any(axis=0)]
        return(all_relationships_dataNEW)

model_package=read_dataset_from_file(
    './projects/am2_project/models/Instrument_classifier_for_error_detection/Instrument_classifier_for_error_detection_25.pickle')
all_human_labelled_data=read_dataset_from_file(
    'projects/am2_project/data/human_labelled_data/all_human_labelled_data/all_human_labelled_data_3.pickle')
data=my_func(model_package['data'])
