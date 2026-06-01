from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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
from projects.am1_project.src.am1_pipeline import (
    run_full_model_generation,
    #run_full_model_generation_with_cross_validation,
    #generate_all_embeddings,
    #remove_single_instances
)

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


transformed_embeddings_sample['item_type']=transformed_embeddings_sample["item_type"].astype("category").cat.codes
transformed_embeddings_sample = transformed_embeddings_sample.dropna(subset=["topic"]).reset_index(drop=True)
y=transformed_embeddings_sample['topic']
X=transformed_embeddings_sample.drop('topic', axis=1)
X=transformed_embeddings_sample.drop('agency_id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test set
    random_state=42,     # for reproducibility
 #   stratify=y           # ensures balanced class proportions
)

#7 Create a logistic regression model, test it, and measure it's accuracy

lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
trainedModel=train_model(lr_model_data,
    selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'agency_id', 'has_categories'])

trainedModel=train_model(lr_model_data,
    selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories'])

# Save the model...

save_versioned_pickle_file(rfc, 'rfc_studies', folder='./projects/am1_project/model')
save_versioned_pickle_file(all_df, 'embeddings_with_two_agency', folder='./projects/am1_project/data')
save_versioned_pickle_file(df, 'trainingDataDf', folder='./projects/am1_project/data')

trained_model = read_dataset_from_file('./projects/am1_project/model/trainedModelAllStudies/trainedModelAllStudies_1.pickle')
trained_model = read_dataset_from_file('./projects/am1_project/model/Logistic Regression Model/Logistic Regression Model_2.pickle')

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

report = classification_report(lr_model_data['y_test'].values, y_pred, target_names=set(lr_model_data['y_test'].values))
print(report)
model_name_version=f"logistic_regression_for_topic_classification_v2"
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

df_usoc_test['Topic']
labels = sorted(df_usoc_test['Topic'].unique())
cm = confusion_matrix(df_usoc_test['Topic'], y_pred, labels=labels)
print(cm)
#target_names=list(set(df_usoc_test['Topic'].values))
for index, topic in enumerate(labels):
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

run_full_model_generation(smoke_test_N=1500)

remember you don't have to generate embeddings all the time, if I'm just
testing features, hyperparams, etc. have a flag which disables the embeddings.
generate the embeddings once, save them,


# this is a hack, as the code that generates embeddings doesn't have the same colnames
# def. perhaps fix this later
transformed_embeddings = transformed_embeddings.rename(columns={
        "text_label": "TextLabel",
        "item_categories": "ItemCategories",
        "item_type": "ItemType",
        "agency_id": "AgencyId",
        "has_categories": "HasCategories",
        "topic": "Topic"
        })
    

a=remove_single_instances(transformed_embeddings[0:N], 'topic'), 
            
df=remove_single_instances(df, topic_column)


generate_all_embeddings()

a=data_preprocessing(transformed_embeddings[0:N], smoke_test_N=N, topic_column='topic')
preprocessed_embeddings=transformed_embeddings
for N in [1500, 2000]:

# Compare times for running experiment for different N, with/without feature reduction
transformed_embeddings = read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_dedup/transformed_embeddings_dedup_1.pickle')
xgb = XGBClassifier(n_estimators=200)
lr_sweeps=LogisticRegression(max_iter=5000, class_weight="balanced")
clf = DecisionTreeClassifier(max_depth=500,
   max_features=None,
   min_samples_split=10,
   splitter="random")
rfc = RandomForestClassifier(
    n_estimators = 100,
    max_depth=200,             # deep trees (unpruned)
    max_features='sqrt',        # number of features per split
    bootstrap=True,             # bootstrap sampling
    random_state=42
)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
gb = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
# Do 80,000 as well
# I paused this at 40000,False.
for N in [1500, 5000, 10000, 20000, 40000]:
    embeddings=pd.DataFrame(transformed_embeddings[0:N])
    for pca_reduce in [True, False]:
        for model in [clf, rfc, ada]:
            if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
            if isinstance(model, XGBClassifier):
                model_type="XGB"
            if isinstance(model, DecisionTreeClassifier):
                model_type="Decision Tree"
            if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
            if isinstance(model, AdaBoostClassifier):
                model_type="Ada Boost"
            if isinstance(model, GradientBoostingClassifier):
                model_type="Gradient Boost"
            run_full_model_generation(smoke_test_N=N,
                model=model,
                pca_feature_reduction=pca_reduce,
                transformed_embeddings=remove_single_instances(embeddings, 'topic'), 
                notes=f"{model_type}, {N} samples. PCA feature reduction: {pca_reduce}")

try lr, xgb with no feature reduction next
I can do it with feature reduction when i have time

run_full_model_generation(smoke_test_N=1500, 
    pca_feature_reduction=True,
    notes="Logistic regression for topic classification",
    model=LogisticRegression(max_iter=1000))
    


run for xgb now.
for N in [len(transformed_embeddings)]:
for N in [500]:
    embeddings=pd.DataFrame(transformed_embeddings[0:N][['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories', 'topic']])
    for pca_reduce in [False]:
        for model in [xgb]:
            if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
            if isinstance(model, XGBClassifier):
                model_type="XGB"
            if isinstance(model, DecisionTreeClassifier):
                model_type="Decision Tree"
            if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
            if isinstance(model, AdaBoostClassifier):
                model_type="Ada Boost"
            if isinstance(model, GradientBoostingClassifier):
                model_type="Gradient Boost"
            run_full_model_generation(smoke_test_N=N,
                model=model,
                pca_feature_reduction=pca_reduce,
                transformed_embeddings=remove_single_instances(embeddings, 'topic'), 
                notes=f"{model_type}, {N} samples. AgencyId: No. PCA feature reduction: {pca_reduce}")


for N in [500]:
    run_full_model_generation_with_cross_validation(smoke_test_N=N, 
        pca_feature_reduction=True, 
        notes=f"Logistic regression for topic classification with {N} samples, PCA feature reduction and cross validation")

matches=read_dataset_from_file('./projects/am1_project/data/model_data_no_leakage/matches/matches_1.pickle')
THIS IS LATEST
model_data=read_dataset_from_file('./projects/am1_project/data/model_data_no_leakage/model_data_with_embeddings_no_data_leakage/model_data_with_embeddings_no_data_leakage_1.pickle')
N=len(model_data['X_train'])
N=1500
for model in [xgb, lr, rfc]:
    for pca_reduce in [True, False]:
        if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
        if isinstance(model, XGBClassifier):
                model_type="XGB"
        if isinstance(model, DecisionTreeClassifier):
                model_type="Decision Tree"
        if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
        if isinstance(model, AdaBoostClassifier):
                model_type="Ada Boost"
        if isinstance(model, GradientBoostingClassifier):
                model_type="Gradient Boost"    
        run_full_model_generation(smoke_test_N=N, 
            model=model,
            model_data=model_data,
            pca_feature_reduction=pca_reduce, 
            notes=f"{model_type}, {len(model_data['X_train'])} samples. AgencyId: No. PCA feature reduction: {pca_reduce}"
            )           

for x in range(1,383):
    print(set((round(all_dummy_embeddings[f'ItemCategories_emb_{x}'],5)==round(d[f'ItemCategories_emb_{x}'],5)).values))
    if len(list(set((round(all_dummy_embeddings[f'ItemCategories_emb_{x}'],5)==round(d[f'ItemCategories_emb_{x}'],5)).values)))>1:
        print(round(all_dummy_embeddings[f'ItemCategories_emb_{x}'],5))
        print(round(c[f'ItemCategories_emb_{x}'],5))


save_versioned_pickle_file(matches, 'matches_nextsteps', folder='./projects/am1_project/data/matches_nextsteps')

training_embeddings_with_text=encode_columns_narrow(lr_model_data['X_train'][0:150], ['TextLabel', 'ItemCategories'])
training_sample=training_embeddings_with_text.iloc[50]
a=model.encode('Whether employer provided pension scheme is a TypeA or TypeB pension (MainQ)')

a=df_usoc_test.iloc[1000]

matches=[]
count=0
for k1, v1 in df_usoc_test.iterrows():
    print(f"Number of test cases so far: {count}")
    count=count+1 
    print(f"Number of test cases that have duplicates/near duplicates in training: {len(matches)}")    
    for k, v in df_usoc.iterrows():   
        #b=model.encode(v['TextLabel_embeddings'])
        cos_sim=cosine_similarity(v1['TextLabel_embeddings'].reshape(1,-1), v['TextLabel_embeddings'].reshape(1,-1))
        if cos_sim>.9:
            print(k1)
            print(k)
            print(f"{v1['TextLabel']}::::::: {v['TextLabel']}: {cos_sim}")
            matches.append((v1['TextLabel'], v['TextLabel'], cos_sim))
	


   cos_sim=cosine_similarity(v['embedding'].reshape(1,-1), training_sample['embedding'].reshape(1,-1))
   

text1="Child 7 learning resources available - Yes, they used freely available resources"
text2="Post-C19: New benefit claims- Carers allowance, Personal independence payments, or Disability Living Allowance"


: [[0.9999999]]

GET TRAINING+TEST BASED ON SWEEPS

with open("./projects/am2_project/config/am2_config.json") as f:
    project_config = json.load(f)
from src.ml_resources import (
    get_latest_versions_of_project_sweeps,
    obtain_items_from_colectica)
training_sweep_items=get_latest_versions_of_project_sweeps(project_config)
am1_data={}
usoc_training_sweeps=[x for x in training_sweep_items if x['agencyId']=='uk.iser.ukhls']
colectica_utility.get_items_in_containing_items(usoc_training_sweeps,
    am1_data,
    "Summary",
    colectica_client.item_code('Question'))
colectica_utility.get_items_in_containing_items(usoc_training_sweeps,
    am1_data,
    "Label",
    colectica_client.item_code('Variable'))
am1_data_new=colectica_utility.get_topics(am1_data)


all_study_series=colectica_client.search_items(C.item_code('Series'),
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True)['Results']

test_sweeps_dict={}
for series in all_study_series:
    series_item_id=[{"AgencyId": series['AgencyId'],
            "Identifier": series['Identifier'],
            "Version": series['Version']
    }]
    print(series_item_id)
    series_training_sweeps=[x for x in training_sweep_items 
        if x['agencyId']==series['AgencyId']]
    all_series_sweeps=C.search_items(C.item_code('Study'),
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=series_item_id)['Results']
    # because some genscot are in heaf
    filtered_sweeps=[x for x in all_series_sweeps if x['AgencyId']==series['AgencyId']]
    sees_study_ids=[{'agencyId': x['AgencyId'], 
        "identifier": x['Identifier'], 
        "version": x['Version']} for x in filtered_sweeps]
    test_sweeps = [x for x in series_study_ids if x not in series_training_sweeps]
    test_sweeps_dict[series['AgencyId']]=test_sweeps

agency='uk.alspac'
len([x for x in test_sweeps_dict[agency]])
len([x for x in training_sweep_items if x['agencyId']==agency])
titles=[]
for x in test_sweeps_dict[agency]:
    y=C.get_item_json(x['agencyId'], x['identifier'])
    titles.append(y["DublinCoreMetadata"]["Title"]["en-GB"])
for x in sorted(titles):
    print(x)
    
save_versioned_pickle_file(am1_data, 'training_sweeps_usoc', folder='./projects/am1_project/data')

all_usoc_ids= [x['Identifier'] for x in all_usoc]

sweep_items=[]
am1_data={}
agency_id='uk.iser.ukhls'
item_types_for_project=[C.item_code('Question'), C.item_code('Variable')]

sweep_items=get_latest_versions_of_project_sweeps(project_config)
for wave, sweep_id in project_config["ItemsForTrainingAndTest"]["Sweeps"][agency_id].items():
            latest_version_of_sweep=C.get_item_json(
                agency_id,
                sweep_id
            )
            sweep_items.append({
                 "agencyId": latest_version_of_sweep['AgencyId'],
                 "identifier": latest_version_of_sweep['Identifier'],
                 "version": latest_version_of_sweep['Version']
             })
colectica_utility.get_items_in_containing_items(sweeps,
        am1_data,
        "Summary",
        colectica_client.item_code('Question'))


usoc_training=C.search_items(item_types_for_project,
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=sweep_items)['Results']

training_ids=[x['Identifier'] for x in usoc_training]
all_usoc_ids= [x['Identifier'] for x in all_usoc]
    



item_types_for_project=[C.item_code('Question'), C.item_code('Variable')]
all_usoc=C.search_items(item_types_for_project,
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=usoc_study)['Results']

items = C.search_items(item_types_for_project,
            ReturnIdentifiersOnly=True,
            MaxResults=0,
            SearchLatestVersion=True,
            SearchSets=sweep)['Results']
    

    
df_training_sweeps=data_preprocessing(training_sweeps_items, None)

def filter_by_topics(df1, df2, col="Topic"):
    return df1[df1[col].isin(df2[col].unique())]
def remove_duplicate_textlabels(df_usoc, df_usoc_test, col="TextLabel"):
    return df_usoc_test[~df_usoc_test[col].isin(df_usoc[col])]

def remove_items_in_both_training_and_test(training_data, test_data, agency):
    items_in_both_training_and_test=set([x for x,v in training_data[agency].items() 
        if x in test_data[agency].keys()])
    #print(items_in_both_training_and_test)
    print(f"Number of items in both training and test: {len(items_in_both_training_and_test)}")    
    print(f"Training data size before: {len(training_data[agency].items())}")
    training_data[agency]={k: v for k, v in training_data[agency].items() if k not in items_in_both_training_and_test}
    print(f"Training data size after: {len(training_data[agency].items())}")
    print(f"Test data size before: {len(test_data[agency].items())}")
    test_data[agency] = {k: v for k, v in test_data[agency].items() if k not in items_in_both_training_and_test}
    print(f"Test data size after: {len(test_data[agency].items())}")

def show_wave_titles(data, agency):
    waves=sorted(set([v['ContainedIn'] for k, v in data[agency].items()]))
    wave_titles=[]   
    for wave in waves:
        a=colectica_client.get_item_json(agency, wave)
        number_items_in_wave=len([v for k, v in data[agency].items() if v['ContainedIn']==wave])
        #print(f"{wave}, {a["DublinCoreMetadata"]["Title"]["en-GB"]}")
        wave_titles.append(f"{a["DublinCoreMetadata"]["Title"]["en-GB"]} ({number_items_in_wave} items): {wave}")
    for title in sorted(wave_titles):
        print(title)

def move_items(source, destination, waves_to_move,agency):
    print("BEFORE MOVING: ")
    print(f"Number of items in source: {len(source[agency].items())}")
    print(f"Number of items in destination: {len(destination[agency].items())}")
    print(f"Total items: {len(source[agency].items())+len(destination[agency].items())}")
    items_to_move={k:v for k,v in source[agency].items() if v['ContainedIn'] in waves_to_move}
    print(f"Number of items to move: {len(items_to_move.items())}")
    destination[agency]=destination[agency] | items_to_move
    source[agency]= {k:v for k, v in source[agency].items() if k not in destination[agency].keys()}
    print("AFTER MOVING: ")
    print(f"Number of items in source: {len(source[agency].items())}")
    print(f"Number of items in destination: {len(destination[agency].items())}")
    print(f"Total items: {len(source[agency].items())+len(destination[agency].items())}")


#agencies = ['uk.cls.bcs70']
#agencies = ['uk.wchads']
#agencies=['uk.lha']
#agencies=['uk.mrcleu-uos.sws'] PREPROCESSING/FILTERING REMOVES TOO MANY
#agencies=['uk.alspac']
#agencies=['uk.cls.nextsteps']
#agencies=['uk.genscot'] NOT ENOUGH SAMPLES FOR SINGLE STUDY TEST
#agencies=['uk.mrcleu-uos.hcs'] NOT ENOUGH SAMPLES FOR SINGLE STUDY TEST
#agencies=['uk.mrcleu-uos.heaf']  NOT ENOUGH SAMPLES FOR SINGLE STUDY TEST
agencies=['uk.whitehall2']
agencies=['uk.cls.ncds'

agencies=['uk.cls.bcs70', 'uk.wchads', 'uk.lha', 'uk.alspac', 'uk.cls.nextsteps', 'uk.whitehall2', 'uk.cls.ncds']
agencies=['uk.cls.bcs70']
training_data={}
test_data={}
for agency in agencies:
    study_training_data = read_dataset_from_file(
    f'./projects/am1_project/data/sweeps/training/training_sweeps_{agency}/training_sweeps_{agency}_1.pickle'
    )
    training_data = training_data | study_training_data
    study_test_data = read_dataset_from_file(
    f'./projects/am1_project/data/sweeps/test/test_sweeps_{agency}/test_sweeps_{agency}_1.pickle'
    )
    test_data = test_data | study_test_data
    

agency='all_studies'
remove_items_in_both_training_and_test(training_data, test_data, agency)
show_wave_titles(training_data, agency)
show_wave_titles(test_data, agency)
waves_to_move=[
    "90a96359-d59e-423a-8ad9-406c1e710871",
    "913c215c-1dc9-4df8-a893-e85890f1af5b",
    "0a53e3b7-e284-4cf0-81c8-0d4c86207682"
    ]
move_items(test_data, training_data, waves_to_move, agency)

move_items(training_data, test_data, waves_to_move, agency)

save_versioned_pickle_file(training_data, 
    f'training_sweeps_{agency}', 
    folder=f'./projects/am1_project/data/final_sweeps/training/')

save_versioned_pickle_file(test_data, 
    f'test_sweeps_{agency}', 
    folder=f'./projects/am1_project/data/final_sweeps/test/')

raw_model_data={"training": training_data, "test": test_data}

embeddings=generate_embeddings(raw_model_data, embeddings_file_name=agency)

save_versioned_pickle_file(embeddings, f"{agency}_model_embeddings", folder=f'./projects/am1_project/data/model_embeddings')

agencies=['uk.wchads', 'uk.cls.bcs70', 'uk.lha', 'uk.alspac', 'uk.cls.nextsteps', 'uk.whitehall2', 'uk.cls.ncds', 'uk.iser.ukhls']
for agency in agencies:
    print(agency)
    embeddings=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
    df_training_embeddings=embeddings['X_train']
    df_training_embeddings['Topic']=embeddings['y_train']
    df_test_embeddings=embeddings['X_test']
    df_test_embeddings['Topic']=embeddings['y_test']
    df_test_embeddings["TextLabel"] = df_test_embeddings["TextLabel"].str.replace("\xa0", " ", regex=False)
    df_test_embeddings=remove_duplicate_textlabels(df_training_embeddings, df_test_embeddings, col='TextLabel')
    df_test_embeddings=filter_by_topics(df_test_embeddings, df_training_embeddings, col="Topic")
    df_training_embeddings=filter_by_topics(df_training_embeddings, df_test_embeddings, col="Topic")
    embeddings['X_train']=df_training_embeddings.drop('Topic', axis=1)
    embeddings['y_train']=df_training_embeddings['Topic']
    embeddings['X_test']=df_test_embeddings.drop('Topic', axis=1)
    embeddings['y_test']=df_test_embeddings['Topic']
    feature_coLumns=['ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    lr=LogisticRegression(max_iter=5000, class_weight="balanced")
    xgb = XGBClassifier(n_estimators=200)
    rfc = RandomForestClassifier(
        n_estimators = 100,
        max_depth=200,             # deep trees (unpruned)
        max_features='sqrt',        # number of features per split
        bootstrap=True,             # bootstrap sampling
        random_state=42
    )
    pca_reducing=[False, True]
    for pca_reduce in pca_reducing:
        for model in [xgb, lr, rfc]:
            if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
            if isinstance(model, XGBClassifier):
                model_type="XGB"
            if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
            run_full_model_generation( 
                model=model,
                model_data=embeddings,
                pca_feature_reduction=pca_reduce,
                notes=f"{agency}, {model_type}, {len(embeddings['X_train'])} samples. AgencyId: No. PCA feature reduction: {pca_reduce}"
                )           

    


model=LogisticRegression(max_iter=5000, class_weight="balanced")

pca_reduce=False
if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
if isinstance(model, XGBClassifier):
                model_type="XGB"
if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
run_full_model_generation( 
            model=model,
            raw_model_data=raw_model_data,
            model_data=None,
            pca_feature_reduction=pca_reduce,
            notes=f"{agency}, {model_type}, {len(raw_model_data['training'][agency])} samples. AgencyId: No. PCA feature reduction: {pca_reduce}"
            )           

df=read_dataset_from_file('./projects/am1_project/data/model_embeddings/model_embeddings_1.pickle')

df_training=convert_dictionary_to_dataframe(training_data)




AGENCY =3 IS VRY GOOD
{0, 3, 5, 6, 7, 11, 12}
agency=0
df_usoc=df_training_sweeps[df_training_sweeps['AgencyId']==agency]
"""
a=[x for x in list(df_usoc['Topic']) if x[0:3]=='116'] 
df_usoc_no_covid = df_usoc[
    ~df_usoc['Topic'].isin(a)
]
df_usoc_test=df_test_sweeps[df_test_sweeps['AgencyId']==agency]
a=[x for x in list(df_usoc_test['Topic']) if x[0:3]=='116'] 
df_usoc_test_no_covid = df_usoc_test[
    ~df_usoc_test['Topic'].isin(a)
]
df_usoc_no_covid=df_usoc_no_covid[~df_usoc['Topic'].isin(
    list(set(df_usoc_no_covid['Topic']) - set(df_usoc_test_no_covid['Topic']))
)]
"""
df_usoc_test=df_test_sweeps[df_test_sweeps['AgencyId']==agency]


df=read_dataset_from_file(
    f'./projects/am1_project/data/{agency}model_embeddings/{agency}model_embeddings_1.pickle'
    )
df_usoc=df['X_train']
df_usoc['Topic']=df['y_train']
df_usoc_test=df['X_test']
df_usoc_test['Topic']=df['y_test']

df_usoc_test["TextLabel"] = df_usoc_test["TextLabel"].str.replace("\xa0", " ", regex=False)
df_usoc_test=remove_duplicate_textlabels(df_usoc, df_usoc_test, col='TextLabel')
df_usoc_test=filter_by_topics(df_usoc_test, df_usoc, col="Topic")
df_usoc=filter_by_topics(df_usoc, df_usoc_test, col="Topic")

feature_columns=['ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    
lr_sweeps=LogisticRegression(max_iter=5000, class_weight="balanced")
lr_sweeps.fit(convert_df_to_ndarray(df_usoc, feature_columns), df_usoc['Topic'])
test=convert_df_to_ndarray(df_usoc_test, feature_columns)
y_pred=lr_sweeps.predict(test)
predictions_with_probabilities=lr_sweeps.predict_proba(test)
prediction_results=calculate_accuracy(lr_sweeps,
            predictions_with_probabilities,
            test,
            df_usoc_test['Topic'].tolist(),
            N=5)


list(set(lr_sweeps.classes_) - set(df_usoc_test['Topic']))
roc_auc_score(df_usoc_test['Topic'], predictions_with_probabilities, multi_class='ovr')

import numpy as np
from sklearn.metrics import roc_auc_score

present_classes = np.unique(df_usoc_test_no_covid['Topic'])

mask = np.isin(lr_sweeps.classes_, present_classes)

filtered_probs = predictions_with_probabilities[:, mask]

filtered_classes = lr_sweeps.classes_[mask]

roc_auc_score(
    df_usoc_test_no_covid['Topic'],
    filtered_probs,
    labels=filtered_classes,
    multi_class='ovr'
)

"""
DEL THIS WHEN SURE UNNEEDED
y_test=read_dataset_from_file('./projects/am1_project/data/delthis/y_test/y_test_5.pickle')
predictions_with_probabilities2=read_dataset_from_file('./projects/am1_project/data/delthis/predictions_with_probabilities/predictions_with_probabilities_5.pickle')
labels=read_dataset_from_file('./projects/am1_project/data/delthis/labels/labels_3.pickle')
trained_model=read_dataset_from_file('./projects/am1_project/data/delthis/trained_model/trained_model_1.pickle')
test_results=read_dataset_from_file('./projects/am1_project/data/delthis/test_results/test_results_1.pickle')
roc_auc_result=roc_auc_score(y_test, 
        predictions_with_probabilities2, 
        multi_class='ovr', 
        labels=trained_model.classes_)
"""