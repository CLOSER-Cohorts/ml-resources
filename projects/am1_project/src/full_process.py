import tracemalloc
import logging
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from src.ml_resources import (
    read_dataset_from_file,
    filter_values_by_length,
    apply_pipeline,
    convert_dictionary_to_dataframe,
    create_model_data_object,
    train_model,
    calculate_accuracy
    )
from projects.am1_project.src.am1_mlflow import register_model_and_metrics
from projects.am1_project.src.utility import convert_df_to_ndarray    
from src.logging.utility import StructuredMessage, setup_logging

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")
raw_data_file = './projects/am1_project/data/all_raw_data/all_raw_data_1.pickle'

def data_preprocessing(raw_data, smoke_test=False):
    # We now run quality control code to filter items which don't meet certain criteria. 
    # E.g. the question summary is too short, the question has fewer than N categories
    # associated with it, the question summary contains text, a question has a set of
    # categories associated with it that are not deemed to have predictive value, e.g. yes/no
    tracemalloc.start()
    start = time.perf_counter()
    filtered_items=filter_values_by_length(raw_data, "TextLabel", 10)
    filtered_items_by_number_of_categories=filter_values_by_length(filtered_items,
        "ItemCategories", 3)
    filtered_items=filtered_items_by_number_of_categories
    # Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc
    df=convert_dictionary_to_dataframe(filtered_items)
    number_of_raw_items=len(df)
    df['HasCategories']=[int(x) for x in (df['ItemCategories']!='').tolist()]
    df["AgencyId"]= df["AgencyId"].map({'uk.iser.ukhls' : 0, 
        'uk.cls.bcs70' : 1, 
        'uk.cls.mcs' : 2, 
        'uk.wchads' : 3, 
        'uk.lha' : 4, 
        'uk.alspac' : 5, 
        'uk.mrcleu-uos.sws' : 6, 
        'uk.cls.nextsteps' : 7, 
        'uk.genscot' : 8, 
        'uk.mrcleu-uos.heaf' : 9, 
        'uk.whitehall2' : 10, 
        'uk.mrcleu-uos.hcs' : 11, 
        'uk.cls.ncds' : 12})
    # Question=a1bb19bd-a24a-4443-8728-a6ad80eb42b8, Variable=683889c6-f74b-4d5e-92ed-908c0a42bb2d
    df["ItemType"] = df["ItemType"].map({"a1bb19bd-a24a-4443-8728-a6ad80eb42b8": 1, 
        "683889c6-f74b-4d5e-92ed-908c0a42bb2d": 0})
    # Now we perform some data cleaning. Resetting the index is important so the pipeline
    # operations will work (the indices have to be continuous numeric values with no gaps)
    df = df.dropna(subset=["Topic"])
    # Save the ids in our data for future use when deciding if there is data available
    # on the Colectica repository that isn't in our training/testing data
    df=df.reset_index(drop=True)
    if smoke_test==True:
        df=df[0:1000]
    # We need to remove items that have topics for which there are less than two instances,
    # in order for the stratified splitting performed by train_test_split
    # to be possible.
    class_proportions = df["Topic"].value_counts()
    items_with_unique_topics=list(class_proportions[class_proportions<2].index)
    df=df[~df['Topic'].isin(items_with_unique_topics)].reset_index(drop=True)
    data_preprocessing_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    logger.info(StructuredMessage(message='Data preprocessing',
        application="am1",
        operation_type="data_preprocessing",
        number_of_raw_items=number_of_raw_items,
        number_of_processed_items=len(df),
        size_of_labels=df['TextLabel'].memory_usage(deep=True),
        size_of_categories=df['ItemCategories'].memory_usage(deep=True),
        current_memory_usage=current,
        peak_memory_usage=peak,
        data_preprocessing_time=data_preprocessing_time
        ))
    return df

def create_model_data(embeddings):
    y=embeddings['topic']
    X=embeddings.drop('topic', axis=1)
    #X=transformed_embeddings_sample.drop('agency_id', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% test set
        random_state=42,     # for reproducibility
        stratify=y           # ensures balanced class proportions
    )
    lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
    return lr_model_data

def test_model(trainedModel, lr_model_data):
    input_feature_list=[]
    X_test=convert_df_to_ndarray(lr_model_data['X_test'])
    """
    for input_feature in ['summary_embeddings', 
        'category_embeddings', 'item_type', 'agency_id', 'has_categories']:
        input_feature_list.append(np.vstack(lr_model_data['X_test'][input_feature]))
        X_test = np.hstack(
        input_feature_list
        )
    """
    y_pred=trainedModel.predict(X_test)
    predictions_with_probabilities=trainedModel.predict_proba(X_test)
    prediction_results=calculate_accuracy(trainedModel,
        predictions_with_probabilities,
        X_test,
        lr_model_data['y_test'].values,
        N=3)
    report = classification_report(lr_model_data['y_test'].values,
        y_pred,
        target_names=set(lr_model_data['y_test'].values),
        output_dict=True)
    return ({"report": report,
            "prediction_results": prediction_results
        })
    
def run_full_model_generation():
    tracemalloc.start()
    print("Read in data...")
    all_raw_data=read_dataset_from_file(raw_data_file)
    print("Preprocess data...")
    df=data_preprocessing(all_raw_data, smoke_test=False)
    df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories', 'Topic']]
    start = time.perf_counter()
    print("Generate embeddings...")
    transformed_embeddings = apply_pipeline(df, ['TextLabel', 'ItemCategories'])
    current, peak = tracemalloc.get_traced_memory()
    embeddings_generation_time = time.perf_counter() - start
    tracemalloc.reset_peak()
    logger.info(StructuredMessage(message='Generate embeddings',
        application="am1",
        operation_type="generate_embeddings",
        number_of_embeddings=len(transformed_embeddings),
        size_of_labels=transformed_embeddings['summary_embeddings'].memory_usage(deep=True),
        size_of_categories=transformed_embeddings['category_embeddings'].memory_usage(deep=True),
        current_memory_usage=current,
        peak_memory_usage=peak,
        embeddings_generation_time=embeddings_generation_time
        ))
    print("Create model data...")    
    lr_model_data=create_model_data(transformed_embeddings)
    start = time.perf_counter()
    print("Train model...")
    trained_model=train_model(lr_model_data)
    model_training_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(StructuredMessage(message='Train model',
        application="am1",
        operation_type="train_am1_model",
        number_of_training_records=len(lr_model_data['X_train']),
        size_of_labels=lr_model_data['X_train']['summary_embeddings'].memory_usage(deep=True),
        size_of_categories=lr_model_data['X_train']['category_embeddings'].memory_usage(deep=True),
        current_memory_usage=current,
        peak_memory_usage=peak,
        model_training_time=embeddings_generation_time
        ))
    print("Test model...")    
    test_results=test_model(trained_model, lr_model_data)
    model_name_version=f"logistic_regression_for_topic_classification"
    notes="Logistic regression for topic classification"
    input_example=lr_model_data['X_train'][:5]
    print("Register model on MLFlow...")
    register_model_and_metrics(trained_model,
        LogisticRegression,
        model_name_version,
        test_results["report"],
        notes,
        input_example,
        raw_data_file,
        test_results["prediction_results"])
