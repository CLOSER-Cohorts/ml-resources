from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import json
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    filter_values_by_length,
    convert_dictionary_to_dataframe,
    apply_pipeline,
    train_model,
    create_model_data_object,
    calculate_accuracy)
from src.ml_resources.data import colectica_utility
from projects.am1_project.api import api_client
from projects.am1_project.src.am1_mlflow import register_model_and_metrics

#PROCESS: 
#1. GET DATA FROM COLECTICA

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py.
# We assume the code is being run from the repository root directory (ml-resources).

am1_data=read_dataset_from_file('./projects/am1_project/data/am1_data_heaf/am1_data_heaf_1.pickle')
study_names=['alspac', 'bcs70', 'genscot', 'hcs', 'heaf', 'lha', 'ncds', 'nextsteps', 'sws', 'usoc', 'wchads', 'whitehall2']
all_raw_data={"all_data": {}}
for study_name in study_names:
    raw_data=read_dataset_from_file(f"./projects/am1_project/data/am1_data_{study_name}/am1_Data_{study_name}_1.pickle")
    for key, subdict in raw_data[list(raw_data.keys())[0]].items():
        subdict["AgencyId"] = list(raw_data.keys())[0]
    all_raw_data["all_data"].update(raw_data[list(raw_data.keys())[0]])
save_versioned_pickle_file(all_raw_data, 'all_raw_data', folder='./projects/am1_project/data')


for key, subdict in raw_data[list(raw_data.keys())[0]].items():
        subdict["AgencyId"] = 'uk.mrcleu-uos.heaf'

# We can check if there is newly available data...
with open("./projects/am1_project/config/am1_config.json") as f:
    project_config = json.load(f)

new_am1_data={}
colectica_utility.get_items_in_containing_items(project_config['Studies'],
    new_am1_data, 
    "Summary",
    colectica_client.item_code('Question'))


question_keys_from_repository_for_dataset = []
for agencyId in new_am1_data.keys():
    question_keys_from_repository_for_dataset.extend(new_am1_data[agencyId].keys())

question_keys_from_current_dataset = []
for agencyId in am1_data.keys():
    question_keys_from_current_dataset.extend(am1_data[agencyId].keys())

new_question_identifiers=check_for_newly_available_data_am1(
    question_keys_from_repository_for_dataset,
    question_keys_from_current_dataset)

# have not yet implemented what to do with new questions
for agency in new_am1_data.keys():
    print(agency)
    print([x for x in new_question_identifiers if x in new_am1_data[agency].keys()])

#2. PERFORM QUALITY CONTROL, E.G. REMOVE DATA WITH MISSING, INADEQUATE values, ARRAYS with
# PARTICULAR VALUES

# We now run quality control code to filter items which don't meet certain criteria. 
# E.g. the question summary is too short, the question has fewer than N categories
# associated with it, the question summary contains text, a question has a set of
# categories associated with it that are not deemed to have predictive value, e.g. yes/no

filtered_questions=filter_values_by_length(am1_data, "TextLabel", 10)

filtered_questions_by_number_of_categories=filter_values_by_length(filtered_questions,
    "ItemCategories", 
    3)

filtered_questions=filtered_questions_by_number_of_categories

#3. Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc

df=convert_dictionary_to_dataframe(filtered_questions)
df['HasCategories']=[int(x) for x in (df['ItemCategories']!='').tolist()]
df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories']]
    

#4. Now we perform some data cleaning. Resetting the index is important so the pipeline
# operations will work (the indices have to be continuous numeric values with no gaps)

df = df.dropna(subset=["Topic"]).reset_index(drop=True)
     
# View the proportion of topics in our data
class_proportions = df["Topic"].value_counts()
print(class_proportions)
print(class_proportions[class_proportions<10])

# We need to remove questions that have topics for which there are less than two instances,
# in order for the stratified splitting performed by train_test_split
# to be possible.

questions_with_unique_topics=list(class_proportions[class_proportions<2].index)
df=df[~df['Topic'].isin(questions_with_unique_topics)].reset_index(drop=True)
df['ItemType']=df["ItemType"].astype("category").cat.codes

# Investigate the proportions of topics at level one. For now we won't do anything
# based on this, but we may use this information at a further point.
level_one_topics=pd.DataFrame([x[0:3] for x in df["Topic"]])
level_one_class_proportions = level_one_topics.value_counts()
print(level_one_class_proportions)

#5. PERFORM DATA TRANSFORMATIONS. TRANSFORM TEXT COLUMNS TO EMBEDDINGS

# If we have already calculated the embeddings and saved them to a pickle file, we read them in...

alspac_raw=read_dataset_from_file('./projects/am1_project/data/am1_data_alspac/am1_Data_alspac_1.pickle')
alspac_embeddings=read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_alspac/transformed_embeddings_alspac_1.pickle')
transformed_embeddings = read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_with_ids/transformed_embeddings_with_ids_1.pickle')

# ...otherwise we can calculate them from scratch, and save them to a file...

transformed_embeddings = apply_pipeline(df, ['TextLabel', 'ItemCategories'])


save_versioned_pickle_file(transformed_embeddings, 'transformed_embeddings_with_ids', folder='./projects/am1_project/data')

#6. split data into training and test

transformed_embeddings_sample=transformed_embeddings
transformed_embeddings_sample['item_type']=transformed_embeddings_sample["item_type"].astype("category").cat.codes
transformed_embeddings_sample = transformed_embeddings_sample.dropna(subset=["topic"]).reset_index(drop=True)
y=transformed_embeddings_sample['topic']
X=transformed_embeddings_sample.drop('topic', axis=1)
X=transformed_embeddings_sample.drop('agency_id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test set
    random_state=42,     # for reproducibility
    stratify=y           # ensures balanced class proportions
)

#7 Create a logistic regression model, test it, and measure it's accuracy

lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
trainedModel=train_model(lr_model_data,
    selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'agency_id', 'has_categories'])

trainedModel=train_model(lr_model_data,
    selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories'])

# Save the model...

save_versioned_pickle_file(trainedModel, 'trainedModel', folder='./projects/am1_project/model')
save_versioned_pickle_file(all_df, 'embeddings_with_two_agency', folder='./projects/am1_project/data')
save_versioned_pickle_file(df, 'trainingDataDf', folder='./projects/am1_project/data')

trained_model = read_dataset_from_file('./projects/am1_project/model/trainedModelAllStudies/trainedModelAllStudies_1.pickle')

input_feature_list=[]
#for input_feature in ['summary_embeddings', 'category_embeddings', 'item_type', 'agency_id', 'has_categories']:
for input_feature in ['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories']:
        input_feature_list.append(np.vstack(lr_model_data['X_test'][input_feature]))
        X_test = np.hstack(
        input_feature_list
)
y_pred=trainedModel.predict(X_test)
predictions_with_probabilities=trainedModel.predict_proba(X_test)
wrong_predictions=calculate_accuracy(trainedModel,
    predictions_with_probabilities,
    X_test,
    lr_model_data['y_test'].values,
    N=3)

#b=obtain_correct_data_labels(final_dataset_copy, "do this", "Flagged")

selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories']
input_feature_list=[]
for input_feature in selected_input_features:
        input_feature_list.append(np.vstack(lr_model_data['X_train'][input_feature]))
X_train = np.hstack(
        input_feature_list
    )
    


model_data_ids = (
    transformed_embeddings['agency_id'] + ':' + transformed_embeddings.index.astype(str)
)

model_data_ids = transformed_embeddings['agency_id']


report = classification_report(lr_model_data['y_test'].values,
    y_pred,
    target_names=set(lr_model_data['y_test'].values),
    output_dict=True)
model_name_version=f"logistic_regression_for_topic_classification"
notes="Logistic regression for topic classification"
input_example=X_train[:5]
register_model_and_metrics(trained_model, 
    LogisticRegression,
    model_name_version,
    report,
    notes,
    input_example,
    "./projects/am1_project/data/am1_data_ncds/am1_data_ncds_1.pickle")

print(report)
training_data=f"Summaries and categories for {len(X_train)} questions"
notes_on_experiment = input("Enter any notes on this experiment you wish to record (e.g. parameters, evaluation metrics, what did/didn't work): ")
with open(f"projects/am1_project/reports/classification_report_topic_classification_v2.txt", "w") as f:
    _ = f.write(f"Classification report for topic classification logistic regression model\n\n")
    _ = f.write(f"Notes: \n{notes_on_experiment}\n\n")
    _ = f.write(f"Model name and version: \n{model_name_version}\n\n")
    _ = f.write(f"Training data: \n{training_data}\n\n")
    _ = f.write(f"Classification report: \n{report}\n\n")
    _ = f.write("\nThese topics have been misclassified.\n\n")
    cm = confusion_matrix(y_test, y_pred)
    target_names=list(set(lr_model_data['y_test'].values))
    for index, topic in enumerate(target_names):
        _ = f.write(f"TOPIC {topic}, {sum(cm[index,])}\n")
        for index2, x in enumerate(cm[index,]):
            if x>0:
                _ = f.write(f"{target_names[index2]}: {x}\n")

    # wrong_predictions may not contain the desired information at present,
    # we may need to update the calculate_accuracy function
    for wrong_prediction in wrong_predictions:
        _ = f.write(wrong_prediction)

cm = confusion_matrix(y_test, y_pred)
print(cm)
target_names=list(set(lr_model_data['y_test'].values))
for index, topic in enumerate(target_names):
    print(f"TOPIC {topic}, {sum(cm[index,])}")
    for index2, x in enumerate(cm[index,]):
        if x>0:
            print(f"{target_names[index2]}: {x}")

for index, x in enumerate(cm):
    print(cm[index,])

all_df=pd.concat([df, df_genscot], ignore_index=True)
all_transformed_embeddings=pd.concat([transformed_embeddings_usoc, transformed_embeddings], ignore_index=True)

all_df=pd.DataFrame()
for x in all_embeddings:
    all_df=pd.concat([all_df, x], ignore_index=True)
    del(x)
    

all_embeddings=[]
for i in range(0,20):
    print(i)
    embeddings=apply_pipeline(usoc_df.iloc[20*2000:], ['TextLabel', 'ItemCategories'])
    all_embeddings.append(embeddings)

genscot_df=read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_genscot/transformed_embeddings_genscot_1.pickle')    
training_df=read_dataset_from_file('./projects/am1_project/data/trainingDataDf/trainingDataDf_1.pickle')

df = read_dataset_from_file('./projects/am1_project/data/pending_training_data/am2_relationships_data_for_future_model/am2_relationships_data_for_future_model_11.pickle')


request_body={
    "TextLabel": ["How often do you play sports?", "Number of drinks per week"],
    "ItemCategories": ["every day every week twice a week", "dog"],
    "ItemType": [1, 0],
    "HasCategories": [1, 1]
}

api_client.execute_query(request_body)
from projects.am1_project.src.full_process import run_full_model_generation