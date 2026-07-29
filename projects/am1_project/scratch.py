from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
    generate_embeddings
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

 THIS CALCULATES confusion_matrix CORRECTLY
X_test=convert_df_to_ndarray(embeddings['X_test'], input_features=feature_columns)
y_pred=model.predict(X_test)    
labels = sorted(embeddings['y_test'].unique())
cm = confusion_matrix(embeddings['y_test'], y_pred, labels=labels)
print(cm)
for index, topic in enumerate(labels):
    print(f"TOPIC {topic}, {sum(cm[index,])}")
    for index2, x in enumerate(cm[index,]):
        if x>0:
            print(f"{labels[index2]}: {x}")

count=0
correct=0
for k,x in embeddings['y_test'].items():
    if x=='10401' and y_pred[count]=='10401':
        print(f"{k}, {count}")
        correct=correct+1
    count=count+1

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
xgb = XGBClassifier(n_estimators=1000, max_depth=10)
lr_sweeps=LogisticRegression(max_iter=5000, class_weight="balanced")
clf = DecisionTreeClassifier(max_depth=500,
   max_features=None,
   min_samples_split=10,
   splitter="random")
rfc = RandomForestClassifier(
    n_estimators = 200,
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

matches_2={}
agencies=['uk.wchads', 'uk.cls.bcs70', 'uk.lha', 'uk.alspac', 'uk.cls.nextsteps', 'uk.whitehall2', 'uk.cls.ncds', 'uk.iser.ukhls']
agencies=['uk.cls.bcs70', 'uk.lha', 'uk.alspac', 'uk.whitehall2']
agencies=['gender']
for agency in agencies:
    #embeddings2=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
    matches_2[agency]=[]
    count=0
    #sample_df = embeddings2['X_train'].sample(n=500)
    #sample_df_2 = embeddings2['X_train'].sample(n=500)
    sample_df=all_embeddings_X_train
    sample_df_2=all_embeddings_X_train
    for k1, v1 in sample_df.iterrows():
        print(f"Number of test cases so far: {count}")
        count=count+1 
        print(f"Number of test cases that have duplicates/near duplicates in training: {len(matches_2[agency])}")    
        for k, v in sample_df_2.iterrows():   
            #b=model.encode(v['TextLabel_embeddings'])
            cos_sim=cosine_similarity(v1['TextLabel_embeddings'].reshape(1,-1), v['TextLabel_embeddings'].reshape(1,-1))
            if cos_sim>.8 and cos_sim<1:
                #print(k1)
                #print(k)
                print(f"{agency}: {v1['TextLabel']}::::::: {v['TextLabel']}: {cos_sim}")
                matches_2[agency].append((v1['TextLabel'], v['TextLabel'], cos_sim))
                break
	

>>> for agency in matches_2.keys(): # >.8, 500 random samples compared to 500 other random samples
   print(f"{agency}: {len(matches_2[agency])}")
...
uk.wchads: 330
uk.cls.bcs70: 136
uk.lha: 151
uk.alspac: 38
uk.cls.nextsteps: 297
uk.whitehall2: 266
uk.cls.ncds: 165
uk.iser.ukhls: 140
>>> for agency in matches.keys(): # >.9, 500 random samples compared to 500 other random samples
   print(f"{agency}: {len(matches[agency])}")
...
uk.wchads: 222
uk.cls.bcs70: 75
uk.lha: 94
uk.alspac: 18
uk.cls.nextsteps: 174
uk.whitehall2: 147
uk.cls.ncds: 130
uk.iser.ukhls: 100
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


all_study_series=colectica_client.search_items(colectica_client.item_code('Series'),
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


def show_wave_titles(data, agency):
    #for k, v in data[agency].items():
    #    print(v)
    #    print(v['ContainedIn'])
    waves=sorted(set([(v['AgencyId'], v['ContainedIn']) for k, v in data[agency].items()]))
    wave_titles=[]   
    for wave in waves:
        a=colectica_client.get_item_json(wave[0], wave[1])
        number_items_in_wave=len([v for k, v in data[agency].items() if v['ContainedIn']==wave[1]])
        #print(f"{wave}, {a["DublinCoreMetadata"]["Title"]["en-GB"]}")
        wave_titles.append(f"{a["DublinCoreMetadata"]["Title"]["en-GB"]} ({number_items_in_wave} items): {wave}")
    for title in sorted(wave_titles):
        print(title)
    #return wave_titles

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
agencies=['uk.iser.ukhls']


all_training_waves=[]
all_test_waves=[]
all_training_data={"all_studies":{}}
all_test_data={"all_studies":{}}

remove_items_in_both_training_and_test(training_data, test_data, agency)
show_wave_titles(training_data, agency)
show_wave_titles(test_data, agency)
waves_to_move=[
    "1c09c1ef-bafc-4e48-acb3-d977072d14b7",
    " 12e575cf-86a5-4a6a-bd47-9f523f6465ca"
    ]
move_items(test_data, training_data, waves_to_move, agency)

move_items(training_data, test_data, waves_to_move, agency)

save_versioned_pickle_file(training_data, 
    f'training_sweeps_{agency}', 
    folder=f'./projects/am1_project/data/final_sweeps/training_no_filtered/')

save_versioned_pickle_file(test_data, 
    f'test_sweeps_{agency}', 
    folder=f'./projects/am1_project/data/final_sweeps/test_no_filtered/')

agency='uk.iser.ukhls'
agency='uk.cls.bcs70'
a=read_dataset_from_file(f'./projects/am1_project/data/final_sweeps/training/training_sweeps_{agency}/training_sweeps_{agency}_1.pickle')
b=read_dataset_from_file(f'./projects/am1_project/data/final_sweeps/test/test_sweeps_{agency}/test_sweeps_{agency}_1.pickle')
    

CODE TO UNDO FILTERS

agencies=['uk.iser.ukhls', 'uk.whitehall2', 'uk.cls.nextsteps', 'uk.lha', 'uk.wchads', 'uk.cls.bcs70', 'uk.alspac', 'uk.mrcleu-uos.sws', 'uk.genscot', 'uk.mrcleu-uos.hcs', 'uk.mrcleu-uos.heaf']
training_data={}
test_data={}
for agency in agencies[1:]:
    training_data = read_dataset_from_file(
    f'./projects/am1_project/data/final_sweeps/training/training_sweeps_{agency}/training_sweeps_{agency}_1.pickle')
    test_data = read_dataset_from_file(
    f'./projects/am1_project/data/final_sweeps/test/test_sweeps_{agency}/test_sweeps_{agency}_1.pickle'
    )
    raw_model_data={"training": training_data, "test": test_data}
    embeddings=generate_embeddings(raw_model_data, 
     embeddings_file_name=f"{agency}",
     embeddings_folder_name="unfiltered_embeddings",
     filter=False)


    save_versioned_pickle_file(embeddings, f"{agency}_model_embeddings", folder=f'./projects/am1_project/data/model_embeddings_not_filtered')

embeddings=read_dataset_from_file('./projects/am1_project/data/model_embeddings_not_filtered/uk.iser.ukhls_model_embeddings/uk.iser.ukhls_model_embeddings_1.pickle')
    

agency='uk.whitehall2'
whitehall2 performance is a lot better so is lha. nextsteps is slightly worse. usoc is slightly worse
big boodst for nshd
agency='uk.cls.nextsteps'
embeddings2=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings_not_filtered/unfiltered_{agency}_model_embeddings/unfiltered_{agency}_model_embeddings_1.pickle')
embeddings=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
        

all_filtered_embeddings={'X_train': pd.DataFrame(),
   'y_train': pd.DataFrame(),
   'X_test': pd.DataFrame(),
   'y_test': pd.DataFrame()}
for agency in agencies:
    print(agency)
    #embeddings=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
    study_embeddings = read_dataset_from_file(f'./projects/am1_project/data/unfiltered_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
    all_filtered_embeddings['X_train']=pd.concat([all_filtered_embeddings['X_train'], 
             study_embeddings['X_train']])
    all_filtered_embeddings['y_train']=pd.concat([all_filtered_embeddings['y_train'], 
             study_embeddings['y_train']])
    all_filtered_embeddings['X_test']=pd.concat([all_filtered_embeddings['X_test'], 
             study_embeddings['X_test']])
    all_filtered_embeddings['y_test']=pd.concat([all_filtered_embeddings['y_test'], 
             study_embeddings['y_test']])

set(y_test) - set(y_train)

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

def split_test_validation_data(all_data):
    test_data=None
    validation_data=None
    test_data=all_data['X_test']
    total_length=len(test_data)
    reduced_test_data=pd.DataFrame()
    reduced_test_data_y=pd.Series()
    validation_data=pd.DataFrame()
    validation_data_y=pd.Series()
    len_validation=0
    for x in set(test_data['ContainedIn']):
        if len_validation + len(test_data[test_data['ContainedIn']==x])<total_length/2:
            validation_data=pd.concat([validation_data, test_data[test_data['ContainedIn']==x]])
            validation_data_y= all_data['y_test'].loc[validation_data.index]
        else:
            reduced_test_data=pd.concat([reduced_test_data, test_data[test_data['ContainedIn']==x]])
            reduced_test_data_y=pd.concat([reduced_test_data_y, 
                all_data['y_test'].loc[reduced_test_data.index]])
    if len(validation_data)==0 or len(validation_data)/len(test_data)<.3:
        validation_data=test_data[0:int(len(test_data)/2)]
        validation_data_y= all_data['y_test'].loc[validation_data.index]
        reduced_test_data=test_data[int(len(test_data)/2):]
        reduced_test_data_y= all_data['y_test'].loc[reduced_test_data.index]
    all_data['X_validation']=validation_data
    all_data['y_validation']=validation_data_y
    all_data['X_test']=reduced_test_data
    all_data['y_test']=reduced_test_data_y


all_embeddings_X_train=pd.DataFrame()
all_embeddings_y_train=pd.DataFrame()
#agencies=['uk.wchads', 'uk.cls.bcs70', 'uk.lha', 'uk.alspac', 'uk.cls.nextsteps', 'uk.whitehall2', 'uk.cls.ncds', 'uk.iser.ukhls']
#agencies=[ 'uk.alspac', 'uk.whitehall2', 'uk.cls.nextsteps', 'uk.cls.bcs70', 'uk.lha', 'uk.wchads', 'uk.iser.ukhls']
agencies=['uk.iser.ukhls', 'uk.whitehall2', 'uk.cls.nextsteps', 'uk.lha', 'uk.wchads', 'uk.cls.bcs70', 'uk.alspac', 'uk.mrcleu-uos.sws', 'uk.genscot', 'uk.mrcleu-uos.hcs', 'uk.mrcleu-uos.heaf']
for agency in agencies:
#if True:
 #   agency='all_studies'
    print(agency)
    #embeddings=read_dataset_from_file(f'./projects/am1_project/data/model_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_1.pickle')
    embeddings=read_dataset_from_file(f'./projects/am1_project/data/unfiltered_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_2.pickle')
   # num_train=len(embeddings['X_train'])
    #num_test=len(embeddings['X_test'])
    #print(f"X_train: {num_train}, X_test: {num_test}. {num_train(num_train + num_test)}")
    #embeddings=all_filtered_embeddings
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
    split_test_validation_data(embeddings)
    gender_topics=embeddings['y_train'][embeddings['y_train']=='10102']
    all_embeddings_X_train=pd.concat([all_embeddings_X_train, embeddings['X_train'].loc[gender_topics.index]])
    all_embeddings_y_train=pd.concat([all_embeddings_y_train, gender_topics])

    print(len(embeddings['X_test']))
    print(len(embeddings['X_validation']))
    texts=pd.concat([embeddings['X_train']['TextLabel'], embeddings['X_test']['TextLabel']])
    categories=pd.concat([embeddings['X_train']['ItemCategories'], embeddings['X_test']['ItemCategories']])
    #label_tdf=vectorizer.fit_transform([str(x) for x in texts.values.tolist()])
    #categories_tdf=vectorizer.fit_transform([str(x) for x in categories.values.tolist()])
    #training_label_tdf=label_tdf[0:len(embeddings['X_train'])]
    #training_categories_tdf=categories_tdf[0:len(embeddings['X_train'])]
    #test_label_tdf=label_tdf[len(embeddings['X_train']):]
    #test_categories_tdf=categories_tdf[len(embeddings['X_train']):]
    #embeddings['X_train']['TextLabel_embeddings']=training_label_tdf.toarray().tolist()
    #embeddings['X_train']['ItemCategories_embeddings']=training_categories_tdf.toarray().tolist()
    #embeddings['X_test']['TextLabel_embeddings']=test_label_tdf.toarray().tolist()
    #embeddings['X_test']['ItemCategories_embeddings']=test_categories_tdf.toarray().tolist()
    #feature_coLumns=['ItemType', 'AgencyId', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    lr=LogisticRegression(max_iter=5000, C=15)
    xgb = XGBClassifier(n_estimators=1000)
    clf = MLPClassifier(random_state=1, max_iter=300)
    rfc = RandomForestClassifier(
        n_estimators = 300,
        max_depth=200,             # deep trees (unpruned)
        max_features='sqrt',        # number of features per split
        bootstrap=True,             # bootstrap sampling
        random_state=42
    )
    #feature_columns=['AgencyId', 'ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    #feature_columns=['ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    feature_columns=['ItemType', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    pca_reducing=[False]
    for pca_reduce in pca_reducing:
        #for model in [lr, rfc, xgb, clf]:
        for model in [lr]:
            if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
            if isinstance(model, XGBClassifier):
                model_type="XGB"
            if isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
            if isinstance(model, MLPClassifier):
                model_type="MLP"
            run_full_model_generation(
                model=model,
                model_data=embeddings,
                pca_feature_reduction=pca_reduce,
                feature_columns=feature_columns,
                notes=f"{agency}, {model_type}, {len(embeddings['X_train'])} samples. C=15. Unfiltered. AgencyId: No. PCA feature reduction: {pca_reduce}. Feature columns: {feature_columns}"
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


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example data
texts = [
    "The product is excellent",
    "Amazing customer service",
    "Very happy with my purchase",
    "Terrible quality and support",
    "I want a refund",
    "Worst experience ever",
    "Great value for money",
    "Highly recommended",
]
>>> len(embeddings['X_train'])
13051
embeddings['X_test']['ItemCategories']

# Labels (1 = positive, 0 = negative)
labels = [1, 1, 1, 0, 0, 0, 1, 1]

# Split raw text first
X_train_text, X_test_text, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Fit TF-IDF on training data only
vectorizer = TfidfVectorizer()
texts=pd.concat([embeddings['X_train']['TextLabel'], embeddings['X_test']['TextLabel']])
categories=pd.concat([embeddings['X_train']['ItemCategories'], embeddings['X_test']['ItemCategories']])
label_tdf=vectorizer.fit_transform([str(x) for x in texts.values.tolist()])
categories_tdf=vectorizer.fit_transform([str(x) for x in categories.values.tolist()])
training_label_tdf=label_tdf[0:len(embeddings['X_train'])]
training_categories_tdf=categories_tdf[0:len(embeddings['X_train'])]
test_label_tdf=label_tdf[len(embeddings['X_train']):]
test_categories_tdf=categories_tdf[len(embeddings['X_train']):]
embeddings['X_train']['TextLabel_embeddings']=training_label_tdf.toarray().tolist()[0:20000]
embeddings['X_train']['ItemCategories_embeddings']=training_categories_tdf.toarray().tolist()[0:20000]
embeddings['X_test']['TextLabel_embeddings']=test_label_tdf.toarray().tolist()
embeddings['X_test']['ItemCategories_embeddings']=test_categories_tdf.toarray().tolist()




X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training matrix shape:", X_train.shape)
print("Test matrix shape:", X_test.shape)

# Train classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

from collections import Counter

my_list = ['a', 'b', 'a', 'c', 'b', 'a']

counts = Counter(my_list)

print(counts)

calculate different levels of top-3-5-7-10

def merge_rare_topics(series, min_count=25):
    # Count occurrences of each topic
    counts = series.value_counts()
    # Topics to merge
    rare_topics = counts[counts < min_count].index
    # Replace rare topics with parent topic
    return series.apply(
        lambda x: str(x)[:-2] if x in rare_topics else x
    )


max_accuracy=0
best_params=None
param_grid={'C': [0.1, 1, 10, 100],
'max_iter': [5000],
'penalty': ['l2'],
'class_weight': ["balanced"]
}
for params in ParameterSampler(param_grid, n_iter=100):
    trained_model = LogisticRegression(**params, )
    print(f"Best so far: {max_accuracy}")
    print(best_params)
    trained_model.fit(X_resampled, y_resampled)
    predictions_with_probabilities=trained_model.predict_proba(X_test)
    y_test=embeddings['y_test']
    prediction_results=calculate_accuracy(trained_model,
            predictions_with_probabilities,
            X_test,
            y_test.tolist(),
            N=5)
    if prediction_results['Accuracy']>max_accuracy:
       max_accuracy=prediction_results['Accuracy']	
       best_params=params

ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(
        model_data['X_train'],
        model_data['y_train'])
    model_data['X_train']=X_resampled
    model_data['y_train']=y_resampled
    

embeddings2=read_dataset_from_file(f'./projects/am1_project/data/transformed_embeddings_with_ids/transformed_embeddings_with_ids_1.pickle')

embeddings2=read_dataset_from_file(f'./projects/am1_project/data/usoc_embeddings/usoc_embeddings_1.pickle')

question_embeddings=embeddings2[embeddings2['item_type']==C.item_code('Question')]
random_vars = var_embeddings.sample(n=len(question_embeddings))

question_var_pairs=pd.DataFrame()


count=0
for k, v in question_embeddings.iterrows():
    print(count)
    count=count+1
    if v['item_type']==C.item_code('Question'):
       matching_item=C.search_relationship_byobject(v['agency_id'], k, item_types=[C.item_code('Variable')])
       if len(matching_item)==1 and matching_item[0]['Item1']['Item1'] in embeddings2.index:
             new_row=pd.concat([v, embeddings2.loc[matching_item[0]['Item1']['Item1']], 
                pd.Series(1)]
                ).to_frame().T 
       else:
             new_row=pd.concat([v, random_vars.iloc[count], 
                pd.Series(0)]
                ).to_frame().T
       new_row.index=[k]
       question_var_pairs=pd.concat([question_var_pairs, new_row])
question_var_pairs.columns=['summary_embeddings_q', 'category_embeddings_q', 'item_type_q', 'topic_q',
       'agency_id_q', 'has_categories_q', 'summary_embeddings_v',
       'category_embeddings_v', 'item_type_v', 'topic_v', 'agency_id_v',
       'has_categories_v', 'valid_question_var_pair']

var_embeddings=embeddings2[embeddings2['item_type']==C.item_code('Variable')]
var_embeddings = var_embeddings.dropna(subset=['topic'])

questions_no_vars=pd.DataFrame()
count=0
for k, v in question_var_pairs[5401:].iterrows():
    print(count)
    #if v['item_type']==C.item_code('Question'):
    new_row=pd.concat([v, random_vars.iloc[count]]).to_frame().T
    new_row.index=[k]
    count=count+1
    questions_no_vars=pd.concat([questions_no_vars, new_row])
questions_no_vars['valid_question_var_pair']=0


question_var_pairs=read_dataset_from_file(f'./projects/am1_project/data/question_var_pairs_usoc/question_var_pairs_usoc_2.pickle')       
question_var_pairs[12]=1

all_training_test_data=pd.concat([question_var_pairs2, questions_no_vars])
idx = all_training_test_data.index.to_list()
np.random.shuffle(idx)
res = all_training_test_data.loc[idx]
res = res.dropna(subset=['topic_q'])
res = res.dropna(subset=['topic_v'])
model=LogisticRegression(max_iter=5000, C=15)

y=res['valid_question_var_pair']
X=res.drop('valid_question_var_pair', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)
model.fit(convert_df_to_ndarray(X_train, ['summary_embeddings_q',
'category_embeddings_q',
'topic_q',
'summary_embeddings_v',
'category_embeddings_v',
'topic_v']), y_train.values.tolist())

model.fit(convert_df_to_ndarray(X_train, ['summary_embeddings_q',
'category_embeddings_q',
'summary_embeddings_v',
'category_embeddings_v']), y_train.values.tolist())


    

X_test_np=convert_df_to_ndarray(X_test, input_features=['summary_embeddings_q',
'category_embeddings_q',
'summary_embeddings_v',
'category_embeddings_v',
])
y_pred=model.predict(X_test_np)
predictions_with_probabilities=model.predict_proba(X_test_np)
"""
prediction_results=calculate_accuracy(model,
            predictions_with_probabilities,
            X_test_np,
            #model_data['y_test'].tolist(),
            y_test.tolist(),
            N=5)
"""
report = classification_report(y_test.values.tolist(),
        y_pred.tolist(),
        #labels=labels,
        #target_names=[str(label) for label in labels],
        output_dict=True)
save_versioned_pickle_file(question_var_pairs, 'question_var_pairs_usoc', folder='./projects/am1_project/data')
save_versioned_pickle_file(questions_no_vars, 'questions_no_vars_usoc', folder='./projects/am1_project/data')

count=0
potential_mismatch=[]
for x in y_pred.tolist():
    if x==0 and y_test.values.tolist()[count]==1 and predictions_with_probabilities[count][0]>.7:
        print(count)
        potential_mismatch.append(y_test.index[count])
        print(y_test.index[count])
    count=count+1
    

no_vars=[]
count=0
for k, v in question_embeddings.iterrows():
    print(count)
    count=count+1
if v['item_type']==C.item_code('Question'):
       matching_item=C.search_relationship_byobject(v['agency_id'], k, item_types=[C.item_code('Variable')])
       if len(matching_item)==0:
           print(k)
           no_vars.append(k)

count=0
input_columns=['summary_embeddings_q', 'category_embeddings_q', 'summary_embeddings_v', 'category_embeddings_v']
vars_to_search=var_embeddings[var_embeddings['topic']==X.loc['05a74e09-7f7b-4909-915d-ff8b9f9f960c']['topic_q']]
vars_to_search=var_embeddings[var_embeddings['topic']=='115']
vars_to_search=X_test
for k, v in vars_to_search.iterrows():
    #print(f"{k}, {count}")
    #new_row=pd.concat([v, random_vars.iloc[count]], ignore_index=True)
    question_part=X_test.iloc[451][['summary_embeddings_q',
       'category_embeddings_q']
       ]
    variable_part=v[['summary_embeddings_v', 'category_embeddings_v']]
    new_row=pd.concat([question_part, variable_part]).to_frame().T
    #new_row=new_row.rename(columns={'summary_embeddings': 'summary_embeddings_v', 'category_embeddings': 'category_embeddings_v'}) 
    print(model.predict(convert_df_to_ndarray(new_row, input_features=input_columns)).item())
    if model.predict(convert_df_to_ndarray(new_row, input_features=input_columns)).item()==1:
        print(k)
       # print(model.predict(convert_df_to_ndarray(new_row, input_features=input_columns)).item())
        print(model.predict_proba(convert_df_to_ndarray(new_row, input_features=input_columns)))
    
    
    questions_no_vars=pd.concat([questions_no_vars, new_row.to_frame().T], ignore_index=True)


questions_no_vars[12]=0


var_
input_columns=[
'summary_embeddings_q',
'category_embeddings_q'
'summary_embeddings_v',
'category_embeddings_v']

a=convert_df_to_ndarray(X_train, ['summary_embeddings_q',
'category_embeddings_q',
'topic_q',
'summary_embeddings_v',
'category_embeddings_v',
'topic_v'])

b=convert_df_to_ndarray(X_test, ['summary_embeddings_q',
'category_embeddings_q',
'topic_q',
'summary_embeddings_v',
'category_embeddings_v',
'topic_v'])

shap_values = explainer(b)

import numpy as np
import pandas as pd

importance = np.abs(shap_values.values).mean(axis=0)

feature_importance = (
    pd.DataFrame({
        "Feature": b.columns,
        "Importance": importance
    })
    .sort_values("Importance", ascending=False)
)

print(feature_importance)