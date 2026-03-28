import json
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    obtain_items_from_colectica,
    check_for_newly_available_data)
from projects.am2_project.src.data.utility import create_am2_input_features, train_semi_supervised_model
from src.ml_resources.data import colectica_utility
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
colectica_client = colectica_utility.C

#item_types_string=["Action", "Archive", "Category", "Category Group", "Category Set", "ClassificationCorrespondenceTable", "ClassificationFamily", "ClassificationIndex", "ClassificationItem", "ClassificationLevel", "ClassificationSeries", "Code List Group", "Code List Set", "Code Set", "Concept", "Concept Group", "Concept Set", "Conceptual Component", "Conceptual Variable", "Conceptual Variable Group", "Conceptual Variable Set", "Conditional", "Control Construct Group", "Control Construct Set", "Data Collection", "Data File", "Data Layout", "DataCollection Methodology", "General Instruction", "Generation Instruction", "Individual", "Instruction Group", "Instrument", "Instrument Group", "Instrument Set", "Interviewer Instruction", "Interviewer Instruction Set", "Logical Product", "Loop", "Managed Representation Group", "Managed Representation Set", "MeasurementItem", "MeasurementConstruct", "Metadata Package", "NCube", "NCube Group", "NCube Set", "Organization", "Organization Group", "Organization Set", "OtherMaterial", "OtherMaterialGroup", "OtherMaterialScheme", "Physical Data Product", "Physical Structure", "PhysicalStructure Set", "Processing Event", "Processing Event Group", "Processing Event Set", "Processing Instruction Group", "Processing Instruction Scheme", "Project", "Quality Standard", "Quality Statement", "Quality Statement Group", "Quality Statement Set", "Question", "Question Activity", "Question Block", "Question Grid", "Question Group", "Question Set", "RecordLayout", "RecordLayout Set", "Repeat", "Represented Variable", "Represented Variable Group", "Represented Variable Set", "Reusable Missing Value", "Sequence", "Series", "Statement", "StatisticalClassification", "Study", "SubSeries", "UnitType", "UnitTypeScheme", "UnitTypeGroup", "Universe", "Universe Group", "Universe Set", "Variable", "Variable Group", "Variable Set", "Variable Statistic", "While"]
#item_types_string=["Data File", "Data Collection", "Variable Statistic", "While", "Instrument"]
#START WITH ONE DATA TYPE, E.G. DATA FILE, BUILD IN THAT
with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)

#TRY ADDING MORE DATA TYPES NOW. CREATE A SERIES OF MODELS, ONE FOR EACH DATA TYPE
#PERHAPS WHEN YOU HAVE MANY MODELS, YOU CAN TRY CREATING A SINGLE DECISION TREE 
#WITH THE CONTENTS OF ALL_DATA

item_types_string = project_config['ItemTypes']

all_relationships_data={}
# maybe automatically get the latest version?
file_path = Path('./projects/am2_project/data/all_am2_relationships_data/all_am2_relationships_data_3.pickle')
if file_path.exists():
    all_relationships_data=read_dataset_from_file(file_path)
else:
    print("File does not exist")

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
        print(f"We already have some data for item type {item_type}")
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
train_semi_supervised_model(dtc,
    df_relationships_unique,
    item_types_string,
    all_training_data,
    all_principal_components=all_principal_components,
    dataset_name="wip",
    generate_classification_report=True)
# dtc was fitted in train_semi_supervised_model so the below check_is_fitted command should
# run and not throw an error.
check_is_fitted(dtc)
save_versioned_pickle_file(all_relationships_data, 
        'all_am2_relationships_data', folder='./projects/am2_project/data')

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
