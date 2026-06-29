This document explains how features for the AM2 project are created and used.

Logic behind transformations

The AM2 project is a set of decision tree models that classifies items as normal or anomalies,
based on their relationships with other items. There are separate training models for 
each item type. The raw data for the models is a Pandas dataframe containing rows representing 
the types of connection patterns that exist between items of that type and other item types.

The decision tree model requires numeric input features. In order to create these input features,
we run a script (check_for_data.py) that checks for data that has not yet been included
in the AM2 training data, and for each new item runs a set of queries to find ancestor
and descendent items.


Source columns and dependencies

The raw input data for the AM2 project is a set of dataframes that are input for the IsolationForest model. There is a dataframe for each item type.

>>> relationships_data_for_training_updated_model.keys()
dict_keys(['Category Set', 'Series', 'Concept', 'Code List Set', 'Data File', 'Instrument', 'OtherMaterial', 'Instrument Set', 'Question Activity', 'Variable Statistic', 'Data Layout', 'Question Set', 'Universe', 'Conditional', 'Statement', 'Interviewer Instruction', 'Question', 'Interviewer Instruction Set', 'Code Set', 'Data Collection', 'Category', 'Control Construct Set', 'Organization', 'Loop', 'Metadata Package', 'Sequence', 'Study', 'Question Group', 'Question Grid', 'Variable', 'Variable Group'])
>>> print(relationships_data_for_training_updated_model['Data File'])
                                                    Category Set  Data Layout  ...  OtherMaterial  ItemType
urn:ddi:uk.iser.ukhls:d0ec6065-65e5-46b9-906d-f...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.iser.ukhls:610a8859-27bb-416e-a96b-f...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.iser.ukhls:20b79b40-6794-4eaa-a3a6-f...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.iser.ukhls:b77f54e4-391f-44cf-b229-f...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.iser.ukhls:03b7de45-ff36-4913-be54-f...           1.0          5.0  ...            0.0       0.0
...                                                          ...          ...  ...            ...       ...
urn:ddi:uk.alspac:dbf7298d-ea8a-432c-9bb0-303ed...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.alspac:319af160-ce6c-481e-be8c-2ff71...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.alspac:c3698197-01c9-4a54-9864-1daf5...           1.0          5.0  ...            1.0       0.0
urn:ddi:uk.alspac:e9873897-4431-4140-b71e-1c66d...           1.0          5.0  ...            0.0       0.0
urn:ddi:uk.alspac:536811a6-b216-40e4-bae7-15d38...           1.0          5.0  ...            0.0       0.0

[520 rows x 17 columns]
>>> relationships_data_for_training_updated_model['Data File'].iloc[0]
Category Set               1.0
Data Layout                5.0
Variable Group             5.0
Variable Statistic         5.0
Code Set                   1.0
Concept                    1.0
Variable                   1.0
Data File                  1.0
Category                   1.0
Study                      5.0
Series                     1.0
Project                    1.0
Interviewer Instruction    0.0
Question                   0.0
Question Grid              0.0
OtherMaterial              0.0
ItemType                   0.0
Name: urn:ddi:uk.iser.ukhls:d0ec6065-65e5-46b9-906d-f8b4ad423a08:3, dtype: float64

These dataframes are input to the train_semi_supervised_model function:

train_semi_supervised_model(
    relationships_data_for_training_updated_model,
    project_config['ItemTypes'],
    dataset_name="wip",
    generate_classification_report=True,
    save_model_in_package_file=True,
    all_models=all_item_models
    )

This function uses an IsolationForest to find outliers in the dataframes (e.g. items that
have unusual connection patterns to other item types).

The train_semi_supervised_model process will ask the human to label the outliers as normal
or anomalous. The output from this process is a dataset where item connection patterns as
labelled as normal or anomalous:

>>> all_item_models['Data File']['data'].iloc[0]
Category Set                     1.0
Data Layout                      5.0
Variable Group                   0.0
Variable Statistic               5.0
Code Set                         1.0
Concept                          0.0
Variable                         1.0
Data File                        1.0
Category                         1.0
Study                            5.0
Series                           1.0
Project                          1.0
Interviewer Instruction          1.0
Question                         1.0
Question Grid                    0.0
OtherMaterial                    0.0
x                          -3.523037
y                           0.019362
DistanceFromOrigin          3.523091
AnomalyScore               -0.097878
ItemType                   Data File
Flagged                           -1
Name: urn:ddi:uk.iser.ukhls:78219a3b-2c6a-4876-8a87-e93f738380d5:3, dtype: object

This data is input training data for a decision tree model. The decision tree model accepts
the following input:

>>> all_item_models['Data File']['model']
DecisionTreeClassifier(class_weight='balanced', max_depth=10)
>>> all_item_models['Data File']['model'].feature_names_in_
array(['Category Set', 'Data Layout', 'Variable Group',
       'Variable Statistic', 'Code Set', 'Concept', 'Variable',
       'Data File', 'Category', 'Study', 'Series', 'Project',
       'Interviewer Instruction', 'Question', 'Question Grid',
       'OtherMaterial', 'ItemType'], dtype=object)
>>> features=all_item_models['Data File']['model'].feature_names_in_
>>> all_item_models['Data File']['data'][features]
                                                    Category Set  Data Layout  Variable Group  ...  Question Grid  OtherMaterial   ItemType
urn:ddi:uk.iser.ukhls:78219a3b-2c6a-4876-8a87-e...           1.0          5.0             0.0  ...            0.0            0.0  Data File
urn:ddi:uk.mrcleu-uos.hcs:331f25b0-d2fe-4704-8e...           0.0          5.0             5.0  ...            0.0            0.0  Data File
urn:ddi:uk.iser.ukhls:94af5e23-8d46-42e9-b466-0...           1.0          5.0             5.0  ...            0.0            0.0  Data File
urn:ddi:uk.iser.ukhls:f09ffea7-9252-4168-a65e-d...           1.0          5.0             0.0  ...            1.0            0.0  Data File
urn:ddi:uk.cls.bcs70:369c1330-4d7c-44b3-9a8e-c4...           1.0          5.0             5.0  ...            0.0            1.0  Data File
urn:ddi:uk.iser.ukhls:907de759-b70c-4ef8-9733-b...           1.0          5.0             0.0  ...            1.0            1.0  Data File
urn:ddi:uk.mrcleu-uos.hcs:02db6551-1da7-4959-b9...           1.0          5.0             5.0  ...            1.0            0.0  Data File
urn:ddi:uk.alspac:6b91d7de-968f-4c74-bf06-5b8fa...           1.0          5.0             5.0  ...            1.0            1.0  Data File
urn:ddi:uk.iser.ukhls:6dddee24-9911-4e23-b5d5-1...           1.0          5.0             5.0  ...            1.0            0.0  Data File
urn:ddi:uk.mrcleu-uos.hcs:5c6f786f-080d-4a81-88...           1.0          5.0             5.0  ...            0.0            0.0  Data File
urn:ddi:uk.iser.ukhls:68af3439-dd6b-4735-969f-3...           1.0          5.0             5.0  ...            1.0            1.0  Data File
urn:ddi:uk.wchads:1d13fc3b-5391-4820-a595-670cf...           1.0          5.0             5.0  ...            1.0            0.0  Data File
urn:ddi:uk.cls.bcs70:0f661fbd-2d34-4245-8553-ff...           1.0          5.0             5.0  ...            0.0            1.0  Data File
urn:ddi:uk.cls.ncds:a737624b-0c24-4e71-90e2-4d5...           1.0          5.0             5.0  ...            1.0            0.0  Data File
urn:ddi:uk.iser.ukhls:b77f54e4-391f-44cf-b229-f...           1.0          5.0             0.0  ...            0.0            0.0  Data File
urn:ddi:uk.iser.ukhls:3327bd22-caf8-4697-8016-e...           1.0          5.0             0.0  ...            0.0            0.0  Data File


Assumptions made during feature design

The decision tree at present accepts an 'ItemType' input feature. As the decision trees are
item specific this isn't used at present but we assume at some future point we may want to train
a single decision tree for all item type. The 'ItemType' column would be useful for this.

At present we assume that relations between an item with other item types can be classified either as:

0.0 There is no connection between these items
1.0 There is a transitive connection between these items (e.g. a references b, b references c, therefore c is transitively referenced by a)
5.0 There a direct connection between these items (e.g. a references b)

A future model may use a more fine-grained measurement of connection (e.g. that represents the distance of a transitive connection).