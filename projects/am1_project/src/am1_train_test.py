from sklearn.linear_model import LogisticRegression
import pandas as pd
from src.ml_resources import (
    read_dataset_from_file,
    calculate_accuracy,
    train_model,
    save_versioned_pickle_file
)
from projects.am1_project.src.utility import (
    filter_by_topics,
    remove_duplicate_textlabels,
    test_model,
    split_test_validation_data,
    create_embeddings
)
from src.dataframe_utility import convert_df_to_ndarray
from sklearn.model_selection import GridSearchCV

feature_columns=['ItemType', 'TextLabel_embeddings', 'ItemCategories_embeddings']
lr_old=LogisticRegression(max_iter=5000, C=1, class_weight="balanced")

# The code here (particularly the cross_validation function) assumes that an
# embeddings object and chronological splits have been created using the code
# in create_chronological_splits. I used the create_chronological_splits
# code to create an embeddings object for 9 studies (genscot and hcs didn't have
# enough data) and then used this embeddings object as input for the cross_validation
# code below, to generate a tuned model that I then tested on a holdout set
# (the holdout was in X_test, but it could just as easily be called X_validation)


all_test_results_cross_val={}
all_trained_models_cross_val={}





if agency_id not in all_trained_models_cross_val.keys():
       all_trained_models_cross_val_confidence[agency_id]=[] 
all_trained_models_cross_val_confidence[agency_id].append(
           max(agency_model.predict_proba(
               [convert_df_to_ndarray(
                   pd.DataFrame(row[sample_feature_columns]).T, 
                   input_features=sample_feature_columns)[0]])[0]).item()
       )

def cross_validation(embeddings):
    waves = sorted(embeddings['X_train']["WaveChronology"].dropna().unique())
    splits = []
    for test_wave in waves[1:]:
        train_idx = np.where(embeddings['X_train']["WaveChronology"] < test_wave)[0]
        test_idx = np.where(embeddings['X_train']["WaveChronology"] == test_wave)[0]
        splits.append((train_idx, test_idx))
    param_grid={
        'C': [0.1, 1, 10, 15, 20, 30, 100],
        'max_iter': [5000],
        'penalty': ['l2'],
        'class_weight': ["balanced"]
        }
    search = GridSearchCV(
        estimator=LogisticRegression(max_iter=5000),
        param_grid=param_grid,
        cv=splits,
        scoring="recall_macro",
        n_jobs=-1,
        refit=True
    )
    X_train=convert_df_to_ndarray(embeddings['X_train'][feature_columns], input_features=feature_columns)
    search.fit(X_train, embeddings['y_train'])

# Test the newly tuned model...
test_results=test_model(search, 
       embeddings,
       feature_columns,
       agency_id,
       all_trained_models_cross_val_confidence)

#...or test a tuned model I've previously saved...
all_trained_models_cross_val_confidence={}
agencies=['uk.iser.ukhls', 'uk.whitehall2', 'uk.cls.nextsteps', 'uk.lha', 'uk.wchads', 'uk.cls.bcs70', 'uk.alspac', 'uk.mrcleu-uos.sws', 'uk.genscot', 'uk.mrcleu-uos.hcs', 'uk.mrcleu-uos.heaf']
for agency_id in agencies:    
   embeddings = create_embeddings(agencies=[agency_id])
   all_trained_models_cross_val=read_dataset_from_file(f'./projects/am1_project/model/cross_val_all_models/cross_val_all_models_2.pickle')
   if agency_id in all_trained_models_cross_val.keys():
       test_results=test_model(all_trained_models_cross_val[agency_id], 
          embeddings,
          feature_columns,
          all_trained_models_cross_val_confidence,
          agency_id=agency_id)
       save_versioned_pickle_file(all_trained_models_cross_val_confidence,
            f"{agency_id}_drift_reference_prediction_probabilities",
            folder=f"./projects/am1_project/data/drift_reference")      
            
                

    all_test_results_cross_val[agency]=test_results
    all_trained_models_cross_val[agency]=search

# Calculate average stats over studies

all_test_results_cross_val=read_dataset_from_file(f'./projects/am1_project/data/cross_val_all_results/cross_val_all_results_3.pickle')
    
mean_recall = sum(
    study["report"]["macro avg"]["recall"]
    for study in all_test_results_cross_val.values()
) / len(all_test_results_cross_val)

mean_precision = sum(
    study["report"]["macro avg"]["precision"]
    for study in all_test_results_cross_val.values()
) / len(all_test_results_cross_val)

mean_top_n_accuracy = sum(
    study["prediction_results"]["TopNAccuracy"]
    for study in all_test_results_cross_val.values()
) / len(all_test_results_cross_val)


