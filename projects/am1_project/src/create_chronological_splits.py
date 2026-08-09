from projects.am1_project.src.utility import (
    filter_by_topics,
    remove_duplicate_textlabels,
    test_model,
    split_test_validation_data,
    create_embeddings
)
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    calculate_accuracy,
    train_model
)
import pandas as pd


# This code isn't starting the data pipeline from scratch, it presumes that data has
# been extracted from the Colectica repository, converted into embeddings and saved
# in files in the 'unfiltered_embeddings' folder


all_studies_embeddings = create_embeddings()
feature_columns=['ItemType', 'TextLabel_embeddings', 'ItemCategories_embeddings']
prediction_model=train_model(all_studies_embeddings, feature_columns)
save_versioned_pickle_file(prediction_model, 'tuned_model', folder='./projects/am1_project/model')    

trained_model = read_dataset_from_file('./projects/am1_project/model/trainedModelAllStudies/trainedModelAllStudies_1.pickle')

all_trained_models_cross_val_confidence={}
test_results=test_model_all_studies(prediction_model, 
       all_studies_embeddings,
       feature_columns,
       all_trained_models_cross_val_confidence=all_trained_models_cross_val_confidence)

#TRY WITH A CUSTOM CROSS-VALIDATION, THIS IS OK FOR STUDY SPECIFIC MODELS?
df=embeddings['X_train']
df2=df[['TextLabel', 'ContainedIn']]
unique_rows = df2[~df2["TextLabel"].duplicated(keep=False)]

unique_rows[unique_rows['ContainedIn']==0]

#df3=embeddings['X_test']
#df4=df3[['TextLabel', 'ContainedIn']]
#unique_rows = df4[~df4["TextLabel"].duplicated(keep=False)]

#for USOC training:
mapping = {
    0: 3,
    1: 2,
    2: 4,
    3: 6,
    4: 1,
    5: 5
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for Whitehall2 training:
mapping = {
    0: 2,
    1: 1,
    2: 7,
    3: 9,
    4: 3,
    5: 8,
    6: 5,
    7: 4,
    8: 6
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for nextsteps training:
mapping = {
    0: 6,
    1: 1,
    2: 5,
    3: 3,
    4: 2,
    5: 4
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for lha/nshd:
mapping = {
    0: 12,
    1: 8,
    2: 14,
    3: 17,
    4: 1,
    5: 16,
    6: 10,
    7: 19,
    8: 23,
    10: 24,
    11: 32,
    12: 2,
    13: 13,
    14: 25,
    15: 1,
    16: 18,
    17: 15,
    18: 25,
    19: 6,
    20: 5,
    21: 22,
    22: 7,
    23: 11,
    24: 20,
    25: 4
    }
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for wchads:
mapping = {
    0: 6,
    1: 1,
    2: 2,
    3: 7,
    4: 5
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for bcs70:

mapping = {
    0: 4,
    1: 2,
    2: 3,
    3: 8,
    4: 6,
    5: 5
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for alspac:
mapping = {
    0: 3,
    1: 2,
    2: 4,
    3: 1
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)
#for sws:
mapping = {
    0: 9,
    1: 3,
    2: 6,
    3: 5,
    4: 1,
    5: 4,
    6: 2,
    7: 7,
    8: 8
}
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)


# IMPORTANT NOTE: HCS ONLY HAS TWO WAVES IN TRAINING: THE FIRST WAVE HAS 13 ITEMS, WHICH
# ALL HAVE THE SAME TOPIC. THEREFORE THERE IS NOT ENOUGH DATA TO PROPERLY TRAIN
# THE MODEL USING CROSS VALIDATION
#
# ALSO THERE IS NO GENSCOT DATA IN THE PICKLE FILE
#for hcs:
mapping = {
    0: 1,
    1: 2
}    
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)

#for heaf:
mapping = {
    0: 1,
    1: 2,
    2: 4,
    3: 3
}    
embeddings['X_train']["WaveChronology"] = embeddings['X_train']["ContainedIn"].map(mapping)


#YP: Whether found it easy or difficult to discuss sexual matter with parents

save_versioned_pickle_file(embeddings, f"{agency}_embeddings_with_chronological_waves", folder='./projects/am1_project/data')
save_versioned_pickle_file(all_test_results_cross_val, f"cross_val_all_results", folder='./projects/am1_project/data')
save_versioned_pickle_file(all_test_results_cross_val, f"cross_val_all_models", folder='./projects/am1_project/model')

waves = sorted(embeddings['X_train']["WaveChronology"].unique())
splits = []
for test_wave in waves[1:]:
    train_idx = np.where(embeddings['X_train']["WaveChronology"] < test_wave)[0]
    test_idx = np.where(embeddings['X_train']["WaveChronology"] == test_wave)[0]
    splits.append((train_idx, test_idx))


