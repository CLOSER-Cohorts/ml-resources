import json
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    obtain_items_from_colectica,
    check_for_newly_available_data)
from projects.am2_project.src.data.utility import create_am2_input_features, train_semi_supervised_model
from src.ml_resources.data import colectica_utility
from projects.am2_project.src.data.utility import *
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
from pathlib import Path
colectica_client = colectica_utility.C

DO UNIVERSE AGAIN
# items_not_to_consider..
#item_types_string=["Action", "Archive", "Category", "Category Group", "Category Set", "ClassificationCorrespondenceTable", "ClassificationFamily", "ClassificationIndex", "ClassificationItem", "ClassificationLevel", "ClassificationSeries", "Code List Group", "Code List Set", "Code Set", "Concept", "Concept Group", "Concept Set", "Conceptual Component", "Conceptual Variable", "Conceptual Variable Group", "Conceptual Variable Set", "Conditional", "Control Construct Group", "Control Construct Set", "Data Collection", "Data File", "Data Layout", "DataCollection Methodology", "General Instruction", "Generation Instruction", "Individual", "Instruction Group", "Instrument", "Instrument Group", "Instrument Set", "Interviewer Instruction", "Interviewer Instruction Set", "Logical Product", "Loop", "Managed Representation Group", "Managed Representation Set", "MeasurementItem", "MeasurementConstruct", "Metadata Package", "NCube", "NCube Group", "NCube Set", "Organization", "Organization Group", "Organization Set", "OtherMaterial", "OtherMaterialGroup", "OtherMaterialScheme", "Physical Data Product", "Physical Structure", "PhysicalStructure Set", "Processing Event", "Processing Event Group", "Processing Event Set", "Processing Instruction Group", "Processing Instruction Scheme", "Project", "Quality Standard", "Quality Statement", "Quality Statement Group", "Quality Statement Set", "Question", "Question Activity", "Question Block", "Question Grid", "Question Group", "Question Set", "RecordLayout", "RecordLayout Set", "Repeat", "Represented Variable", "Represented Variable Group", "Represented Variable Set", "Reusable Missing Value", "Sequence", "Series", "Statement", "StatisticalClassification", "Study", "SubSeries", "UnitType", "UnitTypeScheme", "UnitTypeGroup", "Universe", "Universe Group", "Universe Set", "Variable", "Variable Group", "Variable Set", "Variable Statistic", "While"]
#item_types_string=["Data File", "Data Collection", "Variable Statistic", "While", "Instrument"]
#START WITH ONE DATA TYPE, E.G. DATA FILE, BUILD IN THAT
with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)
item_types_string = project_config['ItemTypes']

#TRY ADDING MORE DATA TYPES NOW. CREATE A SERIES OF MODELS, ONE FOR EACH DATA TYPE
#PERHAPS WHEN YOU HAVE MANY MODELS, YOU CAN TRY CREATING A SINGLE DECISION TREE 
#WITH THE CONTENTS OF ALL_DATA


all_relationships_data={}
# maybe automatically get the latest version?
file_path = Path('./projects/am2_project/data/all_am2_relationships_data/all_am2_relationships_data_82.pickle')
if file_path.exists():
    all_relationships_data=read_dataset_from_file(file_path)
else:
    print("File does not exist")

items_extra=[]
items_extra.append(C.get_item_json('uk.cls.bcs70', '501f2307-f6b5-46f2-9324-fb67384fbeef'))
extra_data=create_am2_input_features(items_extra, C)
501f2307-f6b5-46f2-9324-fb67384fbeef

all_relationships_dataNEW=all_relationships_data.copy()
for item_type in all_relationships_dataNEW.keys():
    #all_relationships_dataNEW[item_type] = all_relationships_dataNEW[item_type].replace({1: 0})
    all_relationships_dataNEW[item_type] = all_relationships_dataNEW[item_type].loc[:, (all_relationships_dataNEW[item_type] != 0).any(axis=0)]
    all_relationships_dataNEW[item_type] = all_relationships_dataNEW[item_type].loc[:, (all_relationships_dataNEW[item_type] != 1).any(axis=0)]
    all_relationships_dataNEW[item_type] = all_relationships_dataNEW[item_type].loc[:, (all_relationships_dataNEW[item_type] != 5).any(axis=0)]

for key in data.columns:
    if key not in extra_data.columns:
        data.loc['uk.cls.bcs70:501f2307-f6b5-46f2-9324-fb67384fbeef', key] = 0
    else:
        data.loc['uk.cls.bcs70:501f2307-f6b5-46f2-9324-fb67384fbeef', key] = float(extra_data[key].values[0])
                #df_relationships.loc[len(df_relationships)] = newRow
                df_relationships.loc[f"{item['AgencyId']}:{item['Identifier']}"] = newRow
                

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
        save_versioned_pickle_file(all_relationships_data, 'all_am2_relationships_data', folder='./projects/am2_project/data')
    else:
        print(f"We already have some data for item type {item_type}")
        df_relationships=all_relationships_data[item_type]    
    # Let's check for newly available data on Colectica...
    ids_for_newly_available_data=check_for_newly_available_data(
        [x['Identifier'] for x in items],
        [x.split(":")[1] for x in df_relationships.index])

    # we can use ids_for_newly_available_data to reduce the size of the input
    # to create_am2_input_features if there are new data available
# need to ask colectica support what these flags for search_items do
# RankResults=True,
 #         ResultOffset=0,
 #       NextResult=0,
 #       ResultOrdering=0,
        

all_training_data={}
all_principal_components={}
# This is the model we will train in semi-supervised fashion... 

for item_type in item_types_string:
        print(f"{item_type},  {len(all_relationships_data[item_type])}")

# check_is_fitted throws an error if the model isn't fitted. The below statement should throw
# an error, but after we run train_semi_supervised_model, check_is_fitted should run and return
# nothing, without throwing an exception.
#check_is_fitted(dtc)
a=train_semi_supervised_model(
    all_relationships_data,
    item_types_string[0:5],
    all_training_data,
    all_principal_components=all_principal_components,
    dataset_name="wip",
    generate_classification_report=True,
    save_model_in_package_file=True)
# dtc was fitted in train_semi_supervised_model so the below check_is_fitted command should
# run and not throw an error.
check_is_fitted(dtc)
    
    
    #save_versioned_pickle_file(all_relationships_data, 'all_am2_relationships_data', folder='./projects/am2_project/data')

# The commands below are useful for reading the supervised model trained by
# train_semi_supervised_model above into memory for inspection
all_relationships_dataNEW=all_relationships_data.copy()

def my_func(all_relationships_dataNEW):
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 0).any(axis=0)]
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 1).any(axis=0)]
        all_relationships_dataNEW = all_relationships_dataNEW.loc[:, (all_relationships_dataNEW != 5).any(axis=0)]
        return(all_relationships_dataNEW)

all_labelled_data=pd.DataFrame()
model_package=read_dataset_from_file(
    './projects/am2_project/models/Instrument_classifier_for_error_detection/Instrument_classifier_for_error_detection_25.pickle')
data=my_func(model_package['data'])
data['ItemType']=0
all_labelled_data=pd.concat([all_labelled_data, data])
model_package=read_dataset_from_file(
    './projects/am2_project/models/Series_classifier_for_error_detection/Series_classifier_for_error_detection_38.pickle')
#data=model_package['data']
#data.loc['uk.cls.mcs:0d8a7220-c61b-4542-967d-a40cb5aca430']=1
data=my_func(model_package['data'])
data['ItemType']=1
all_labelled_data=pd.concat([all_labelled_data, data])
model_package=read_dataset_from_file(
    './projects/am2_project/models/Code Set_classifier_for_error_detection/Code Set_classifier_for_error_detection_6.pickle')
data=my_func(model_package['data'])
data['ItemType']=2
all_labelled_data=pd.concat([all_labelled_data, data])
model_package=read_dataset_from_file(
    './projects/am2_project/models/Question_classifier_for_error_detection/Question_classifier_for_error_detection_2.pickle')
data=my_func(model_package['data'])
data['ItemType']=3
all_labelled_data=pd.concat([all_labelled_data, data])
model_package=read_dataset_from_file(
    './projects/am2_project/models/Variable_classifier_for_error_detection/Variable_classifier_for_error_detection_2.pickle')
data=my_func(model_package['data'])
data['ItemType']=4
all_labelled_data=pd.concat([all_labelled_data, data])
model_package=read_dataset_from_file(
    './projects/am2_project/models/Data File_classifier_for_error_detection/Data File_classifier_for_error_detection_2.pickle')
data=my_func(model_package['data'])
data['ItemType']=5
all_labelled_data=pd.concat([all_labelled_data, data])



dtc=model_package_code_set['model']
plt.figure(figsize=(10, 10))
tree.plot_tree(
    dtc2,
    class_names=["2", "3"],
    filled=True,    
)
plt.show()
data=model_package['data']
data.columns[5]

data=all_labelled_data
X=data.drop(columns=feature_names)
y=data['Flagged']
X = X.replace({np.nan: 0})
model = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
dtc2 = DecisionTreeClassifier(class_weight='balanced')
dtc2.fit(X,y)


feature_names=['x', 'y', 'DistanceFromOrigin', 'AnomalyScore','Flagged']

all_training_data_df=pd.DataFrame()
count=0
for item_type in all_labelled_datall_labelled_data.keys():
    all_labelled_data[item_type]['ItemType']=count
    all_training_data_df=pd.concat([all_training_data_df, all_labelled_data[item_type]])
    count=count+1



X=all_training_data_df.drop(columns=['Flagged'])
y=all_training_data_df['Flagged']

try:
    response = client.chat_postMessage(
        channel="SECRET",  # or channel ID like "C123456"
        text="Hello from a ML pipeline 👋 This is a test, please ignore!" 
    )
    print("Message sent:", response["ts"])
except SlackApiError as e:
    print("Error:", e.response["error"])