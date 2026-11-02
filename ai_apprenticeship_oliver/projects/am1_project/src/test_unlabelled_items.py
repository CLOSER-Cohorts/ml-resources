from src.ml_resources.data import colectica_utility
from evidently import Report
from evidently.metrics import ValueDrift
from src.ml_resources import (
            read_dataset_from_file,
            get_max_file_version
        )
from projects.am1_project.src.utility import convert_df_to_ndarray
import pandas as pd
from pathlib import Path
import os

colectica_client = colectica_utility.C

os.chdir("../../working_dir/ml-resources")
transformed_embeddings = read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_with_ids/transformed_embeddings_with_ids_1.pickle')
items_with_no_topics=transformed_embeddings[transformed_embeddings["topic"].isna()]
all_trained_models_cross_val=read_dataset_from_file(f'./projects/am1_project/model/cross_val_all_models/cross_val_all_models_2.pickle')

# get counts per study

items_with_no_topics["agency_id"].value_counts()

# create a df with 100 random rows from each study

sampled_df = (
    items_with_no_topics
    .groupby("agency_id", group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 100), random_state=42))
)

# make item type categorical

sampled_df["item_type"] = sampled_df["item_type"].replace({
    "683889c6-f74b-4d5e-92ed-908c0a42bb2d": 0,
    "a1bb19bd-a24a-4443-8728-a6ad80eb42b8": 1
})

items_with_no_topics_ids=items_with_no_topics.index


# YOU NEED TO MOVE THE BELOW CODE INTO THE API SO IT'S GENERATING
# DATA FOR DATA DRIFT ANALYSIS.
#
# YOU NEED TO GENERATE THE REFERENCE DATASET FOR EACH DATASET AS WELL,
# AND PERFORM THE DRIFT ANALYSIS IN THE API AND LOG IT
#
# AFTER THAT, GO THROUGH THE PREDICTIONS

sample_feature_columns=['item_type', 'summary_embeddings', 'category_embeddings']
labels={}
#agency_probabilities={}
rows = []
model_probabilities={}
for index, row in sampled_df.iterrows():
    agency_id=row['agency_id']
    if agency_id in all_trained_models_cross_val.keys(): 
       item_type=row['item_type']
           #print(index, agency_id, item_type)
       agency_model=all_trained_models_cross_val[agency_id]
       item_details=colectica_client.get_item_json(agency_id, index)
       if item_type==0:
          label=item_details['Label']['en-GB']
       else:
          label=item_details['Summary']['en-GB']
       labels[index]=label
       prediction=agency_model.predict([convert_df_to_ndarray(
          pd.DataFrame(row[sample_feature_columns]).T, 
          input_features=sample_feature_columns)[0]])[0]
       if agency_id not in model_probabilities.keys():
          model_probabilities[agency_id]={}
       if prediction not in model_probabilities[agency_id]:
          model_probabilities[agency_id][prediction]=[] 
       model_probabilities[agency_id][prediction].append(
           max(agency_model.predict_proba(
               [convert_df_to_ndarray(
                   pd.DataFrame(row[sample_feature_columns]).T, 
                   input_features=sample_feature_columns)[0]])[0]).item()
       ) 
       rows.append({
        "Study": agency_id,
        "Index": index,
        "Label": label,
        "Prediction": prediction,
        "Actual topic": None
       })
       print(f"Label: {label}, study: {agency_id}, index: {index}, prediction: {prediction}")

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["Study", "Label"], keep="first")
#sampled_df.loc[df['Index']].to_csv("predictions.csv", index=False)
df.to_csv("predictions.csv", index=False)



metrics=[]
metrics.extend([
                ValueDrift(column='prediction_probabilities', method="psi"),
                ValueDrift(column='prediction_probabilities', method="ks")])
report = Report(
                        metrics=metrics
            )

#model_probabilities=all_trained_models_cross_val_confidence[agency_id]
folder = f"./projects/am1_project/data/drift_reference/{agency_id}_drift_reference"
object_name = f"{agency_id}_drift_reference"
#current_file_version=1
max_file_version = get_max_file_version(Path(f"{folder}"), object_name)
file_path = Path(f"{folder}/{object_name}_{max_file_version}.pickle")
X_reference = read_dataset_from_file(file_path)

training_probabilities_usoc = all_trained_models_cross_val_confidence[agency_id]

topic='111'
model_probabilities[agency_id][topic]
all_trained_models_cross_val_confidence[agency_id][topic]

snapshot = report.run(
    reference_data=pd.DataFrame({
       "prediction_probabilities" : training_probabilities_usoc
       }),
    current_data=pd.DataFrame({
       "prediction_probabilities" : model_probabilities['uk.iser.ukhls']
       })
    )        

for x in snapshot.dict()['metrics']:
    metric=x['config']['method']
    print(f"{metric}: {x['value']}")

    drift_present_psi=False
    drift_present_chi_square=False
    if metric=='psi':
                            drift_present_psi=x['value'] > project_config['Thresholds'][item_type]['Psi']#x['config']['threshold']
                        elif metric=='chisquare':
                            drift_present_chi_square=x['value'] < project_config['Thresholds'][item_type]['ChiSquare'] # x['config']['threshold']
                    

for column in all_item_models[item_type]['data'].columns:
    if column in X_input.columns and not are_series_the_same(X_reference[column], 
        X_input[column]):
            



    
                



agency_model.predict([convert_df_to_ndarray(
       pd.DataFrame(row[sample_feature_columns]).T, 
       input_features=sample_feature_columns)[0]])

agency_model.predict([convert_df_to_ndarray(
       sampled_df[sample_feature_columns], 
       input_features=sample_feature_columns)[576]])

agency_model.predict([convert_df_to_ndarray(pd.DataFrame(row[sample_feature_columns]).T, input_features=sample_feature_columns)[0]])[0]

iterate through labels. run model.encode for each label, run it through
the prediction model. store the models answer and my human-generated label
run classification report, to get performance metrics

Implement the below: i need to run the test_model for each model again, 
and store the probabilities as a refrence dataset

I created a reference set of confidence scores generated by the model on test 
data, and sets of confidence scores generated by the deployed model. I measured 
data drift using the Population Stability Index (PSI) and 
Kolmogorov-Smirnov (KS) statistics.

the probabilities generated from testing a batch of data can serve as
a test dataset

# 0 is Variable item type, 1 is Question item type
agency_model=all_trained_models_cross_val['uk.iser.ukhls']
feature_cols_2=['item_type', 'summary_embeddings', 'category_embeddings']
agency_model.predict([convert_df_to_ndarray(
    sampled_df[feature_cols_2], 
    input_features=feature_cols_2)[576]])


import numpy as np

target = a

result = transformed_embeddings[
    transformed_embeddings["summary_embeddings"].apply(
        lambda x: np.array_equal(x, target)
    )
]