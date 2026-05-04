from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import random
from src.ml_resources import (
    read_dataset_from_file,
    create_model_data_object,
    train_model,
    calculate_accuracy
)
from projects.am1_project.src.utility import get_item_from_topic_name
from src.ml_resources.data import colectica_utility

colectica_client = colectica_utility.C

def generate_prediction_input(test_data):
    input_feature_list=[]
    for input_feature in ['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories']:
        input_feature_list.append(np.vstack(test_data[input_feature]))
        X_test = np.hstack(
        input_feature_list
        )
    return X_test    

def generate_model_data(embeddings, test_size=0.2):
    embeddings['item_type']=embeddings["item_type"].astype("category").cat.codes
    embeddings = embeddings.dropna(subset=["topic"])
    y=embeddings['topic']
    X=embeddings.drop('topic', axis=1)
    X=embeddings.drop('agency_id', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,      
        random_state=42,     # for reproducibility
        stratify=y           # ensures balanced class proportions
    )
    lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
    return lr_model_data


def test_for_performance_on_sensitive_topics(topic, items=None):
    # This is the details for the containing item on the staging server. The
    # version may be different on production/dev environments, the code
    # won't work if the version isn't as specified below
    if items is None:
        print("Get all sensitive topics...")
        containing_item={"AgencyId": "uk.closer",
	        "Identifier": "81138b12-b89d-4cc5-8b36-42eff9b2639b",
	        "Version": 1324
        }
        sensitive_question_groups=get_item_from_topic_name(topic,
                    colectica_client.item_code('Question Group'),
                    containing_item,
                    colectica_client)
        sensitive_variable_groups=get_item_from_topic_name(topic,
                    colectica_client.item_code('Variable Group'),
                    containing_item,
                    colectica_client)
        all_sensitive_groups=sensitive_question_groups+sensitive_variable_groups              
        items=[]
        count=0
        for group in all_sensitive_groups:
	        print(count)
	        count=count+1
	        results=colectica_client.search_relationship_bysubject(group['AgencyId'], 
                group['Identifier'], 
                Version=group['Version'], 
	        item_types=[colectica_client.item_code("Variable"), colectica_client.item_code("Question")],
	        Descriptions=True)
	        items.extend(results)
    # You should probably save items to a pickle file to speed things up
    print("Start creating test set for sensitive items...")
    transformed_embeddings = read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_with_ids/transformed_embeddings_with_ids_1.pickle')
    sensitive_topic_ids=[x['Identifier'] for x in items]
    sensitive_topic_embedding_ids=[x for x in sensitive_topic_ids if x in transformed_embeddings.index]
    """
    FUTURE WORK: get embeddings for items with sensitive topics that are not in 
    transformed_embeddings
    input_for_embeddings=[x for x in items if x['Identifier'] not in sensitive_topic_embedding_ids]
    raw_data={}
    for item in input_for_embeddings:
        if item['ItemType']==colectica_client.item_code('Question'):
            label_field="Summary"
        else:
            label_field="Label"
        colectica_utility.get_item_text(item['ItemType'], 
            label_field,
            raw_data,
            study_items=[item])
        colectica_utility.get_categories_for_items(item['AgencyId'], list(raw_data[item['AgencyId']].keys()), all_items=raw_data, verbose=True)
    """
    shuffled = sensitive_topic_embedding_ids.copy()
    random.shuffle(shuffled)
    half_way=int(len(shuffled)*.85)
    training_ids=shuffled[:half_way]
    test_ids=shuffled[half_way:]
    sensitive_transformed_embeddings=transformed_embeddings.loc[test_ids]
    train_test_embeddings = transformed_embeddings[~transformed_embeddings.index.isin(test_ids)]
    lr_model_data=generate_model_data(train_test_embeddings)
    print("Training model...")
    trainedModel=train_model(lr_model_data,
        selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories'])
    # Test general performance
    X_test=generate_prediction_input(lr_model_data['X_test'])
    y_pred=trainedModel.predict(X_test)
    predictions_with_probabilities=trainedModel.predict_proba(X_test)
    wrong_predictions=calculate_accuracy(trainedModel,
        predictions_with_probabilities,
        X_test,
        lr_model_data['y_test'].values,
        N=3)
    # Test performance on sensitive topic
    lr_model_data=generate_model_data(sensitive_transformed_embeddings, test_size=.99)
    X_test=generate_prediction_input(lr_model_data['X_test'])
    y_pred=trainedModel.predict(X_test)
    predictions_with_probabilities=trainedModel.predict_proba(X_test)
    wrong_predictions=calculate_accuracy(trainedModel,
        predictions_with_probabilities,
        X_test,
        lr_model_data['y_test'].values,
        N=3)
    return {
        "SensitiveItems": items,
        "SensitivePredictions": y_pred,
        "ModelData": lr_model_data,
        "SensitiveLabels": lr_model_data['y_test'].values,
        "SensitiveEmbeddings": sensitive_transformed_embeddings,
        "TrainedModel": trainedModel,
        "WrongPredictions": wrong_predictions
    }

save_versioned_pickle_file(items, "sensitive_10103_ethnic_items", folder='./projects/am1_project/data')

age_results3=test_for_performance_on_sensitive_topics('10107')
gender_results=test_for_performance_on_sensitive_topics('10102')
results=test_for_performance_on_sensitive_topics('10103', items)

for i, x in enumerate(age_results["SensitivePredictions"]):
	    print(f"{i}, {x}")

[x for x in age_results["SensitiveItems"] if x['Identifier']==age_results["ModelData"]["X_test"].index[460]]

transformed_embeddings['item_type']=transformed_embeddings["item_type"].astype("category").cat.codes
y=transformed_embeddings['topic']
X=transformed_embeddings.drop('topic', axis=1)
X=transformed_embeddings.drop('agency_id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.35,       # 20% test set
    random_state=42,     # for reproducibility
    stratify=y           # ensures balanced class proportions
)

lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
trainedModel=train_model(lr_model_data,
    selected_input_features=['summary_embeddings', 'category_embeddings', 'item_type', 'has_categories'])

input_feature_list=[]
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

report_bias = classification_report(lr_model_data['y_test'].values, 
    y_pred,
    target_names=set(lr_model_data['y_test'].values),
    output_dict=True)

save_versioned_pickle_file(report_bias, "report_bias", folder='./projects/am1_project/data')
