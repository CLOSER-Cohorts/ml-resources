import json
from src.ml_resources import read_dataset_from_file, obtain_items_from_colectica
from projects.am2_project.src.data.utility import create_am2_input_features, train_semi_supervised_model
from src.ml_resources.data import colectica_utility
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted
colectica_client = colectica_utility.C

#item_types_string=['Action', 'Archive', 'Category', 'Category Group', 'Category Set', 'ClassificationCorrespondenceTable', 'ClassificationFamily', 'ClassificationIndex', 'ClassificationItem', 'ClassificationLevel', 'ClassificationSeries', 'Code List Group', 'Code List Set', 'Code Set', 'Concept', 'Concept Group', 'Concept Set', 'Conceptual Component', 'Conceptual Variable', 'Conceptual Variable Group', 'Conceptual Variable Set', 'Conditional', 'Control Construct Group', 'Control Construct Set', 'Data Collection', 'Data File', 'Data Layout', 'DataCollection Methodology', 'General Instruction', 'Generation Instruction', 'Individual', 'Instruction Group', 'Instrument', 'Instrument Group', 'Instrument Set', 'Interviewer Instruction', 'Interviewer Instruction Set', 'Logical Product', 'Loop', 'Managed Representation Group', 'Managed Representation Set', 'MeasurementItem', 'MeasurementConstruct', 'Metadata Package', 'NCube', 'NCube Group', 'NCube Set', 'Organization', 'Organization Group', 'Organization Set', 'OtherMaterial', 'OtherMaterialGroup', 'OtherMaterialScheme', 'Physical Data Product', 'Physical Structure', 'PhysicalStructure Set', 'Processing Event', 'Processing Event Group', 'Processing Event Set', 'Processing Instruction Group', 'Processing Instruction Scheme', 'Project', 'Quality Standard', 'Quality Statement', 'Quality Statement Group', 'Quality Statement Set', 'Question', 'Question Activity', 'Question Block', 'Question Grid', 'Question Group', 'Question Set', 'RecordLayout', 'RecordLayout Set', 'Repeat', 'Represented Variable', 'Represented Variable Group', 'Represented Variable Set', 'Reusable Missing Value', 'Sequence', 'Series', 'Statement', 'StatisticalClassification', 'Study', 'SubSeries', 'UnitType', 'UnitTypeScheme', 'UnitTypeGroup', 'Universe', 'Universe Group', 'Universe Set', 'Variable', 'Variable Group', 'Variable Set', 'Variable Statistic', 'While']
#item_types_string=['Data File', 'Data Collection', 'Variable Statistic', 'While', 'Instrument']
#START WITH ONE DATA TYPE, E.G. DATA FILE, BUILD IN THAT
with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)

item_types_string = project_config['ItemTypes']

items=obtain_items_from_colectica(item_types_string)

#df_relationships=create_am2_input_features(items, colectica_client)
df_relationships=read_dataset_from_file('./projects/am2_project/data/Instrument_Relationships/Instrument_Relationships_1.pickle')
df_relationships_unique=df_relationships.drop_duplicates().fillna(0)
# This is the model we will train in semi-supervised fashion...
dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
# check_is_fitted throws an error if the model isn't fitted. The below statement should throw
# an error, but after we run train_semi_supervised_model, check_is_fitted should run and return
# nothing, without throwing an exception.
check_is_fitted(dtc)
all_data={}
all_principal_components={}
train_semi_supervised_model(dtc,
    df_relationships_unique,
    item_types_string,
    all_data,
    all_principal_components=all_principal_components,
    dataset_name="wip",
    generate_classification_report=True)
# dtc was fitted in train_semi_supervised_model so the below check_is_fitted command should
# run and not throw an error.
check_is_fitted(dtc)


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
