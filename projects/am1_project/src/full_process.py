import tracemalloc
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import top_k_accuracy_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from xgboost import XGBClassifier
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    filter_values_by_length,
    apply_pipeline,
    convert_dictionary_to_dataframe,
    create_model_data_object,
    train_model,
    calculate_accuracy
    )
from projects.am1_project.src.am1_mlflow import (
    register_model_and_metrics,
    register_model_and_cross_validation_metrics
    )
from projects.am1_project.src.utility import convert_df_to_ndarray    
from src.logging.utility import StructuredMessage, setup_logging

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")
raw_data_filename = './projects/am1_project/data/all_raw_data/all_raw_data_1.pickle'

# We need to filter that JSON dict because multiple categories in a dataframe row
# are too messy to deal with
def filter_items(raw_data):
    filtered_items=filter_values_by_length(raw_data, "TextLabel", 10)
    filtered_items_by_number_of_categories=filter_values_by_length(filtered_items,
        "ItemCategories", 3)
    filtered_items=filtered_items_by_number_of_categories
    return filtered_items

def preprocess_data_frame(df, 
    item_categories_col='ItemCategories',
    has_categories_col='HasCategories', 
    agency_col='AgencyId',
    item_type_col='ItemType', 
    topic_col='Topic'):
    if pd.api.types.is_string_dtype(df[item_categories_col]):
        df[has_categories_col]=[int(x) for x in (df[item_categories_col]!='').tolist()]
    df[agency_col]= df[agency_col].map({'uk.iser.ukhls' : 0, 
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
    df[item_type_col] = df[item_type_col].map({"a1bb19bd-a24a-4443-8728-a6ad80eb42b8": 1, 
        "683889c6-f74b-4d5e-92ed-908c0a42bb2d": 0})
    # Now we perform some data cleaning. Resetting the index is important so the pipeline
    # operations will work (the indices have to be continuous numeric values with no gaps)
    df = df.dropna(subset=[topic_col])
    # Save the ids in our data for future use when deciding if there is data available
    # on the Colectica repository that isn't in our training/testing data
    df=df.reset_index(drop=True)
    return df

def remove_single_instances(df, topic_column="Topic"):
    class_proportions = df[topic_column].value_counts()
    items_with_unique_topics=list(class_proportions[class_proportions<2].index)
    df=df[~df[topic_column].isin(items_with_unique_topics)].reset_index(drop=True)
    return df
    

def data_preprocessing(raw_data, smoke_test_N=None, topic_column="Topic"):
    # We now run quality control code to filter items which don't meet certain criteria. 
    # E.g. the question summary is too short, the question has fewer than N categories
    # associated with it, the question summary contains text, a question has a set of
    # categories associated with it that are not deemed to have predictive value, e.g. yes/no
    tracemalloc.start()
    start = time.perf_counter()
    filtered_items=filter_items(raw_data)
    # Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc
    df=convert_dictionary_to_dataframe(filtered_items)
    df=preprocess_data_frame(df)
    if smoke_test_N!=None:
        df=df[0:smoke_test_N]
    # We need to remove items that have topics for which there are less than two instances,
    # in order for the stratified splitting performed by train_test_split
    # to be possible.
    df=remove_single_instances(df, topic_column)
    number_of_raw_items=len(df)
    data_preprocessing_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
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

def create_model_data(embeddings, pca_feature_reduction=False, model=None):
    y=embeddings['topic']
    X=embeddings.drop('topic', axis=1)
    if isinstance(model, XGBClassifier):
        le = LabelEncoder()
        y = le.fit_transform(y)
    #X=transformed_embeddings_sample.drop('agency_id', axis=1)
    if pca_feature_reduction:
        pca = PCA(n_components=128)
        reduced_summary_embeddings = pca.fit_transform(convert_df_to_ndarray(
            X, input_features=['summary_embeddings']))
        reduced_category_embeddings = pca.fit_transform(convert_df_to_ndarray(
            X, input_features=['category_embeddings']))
        X['summary_embeddings']=reduced_summary_embeddings.tolist()
        X['category_embeddings']=reduced_summary_embeddings.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% test set
        random_state=42,     # for reproducibility
        stratify=y           # ensures balanced class proportions
    )
    lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
    return lr_model_data

def test_model(trained_model, lr_model_data):
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
    y_pred=trained_model.predict(X_test)
    predictions_with_probabilities=trained_model.predict_proba(X_test)
    prediction_results = None
    if isinstance(trained_model, LogisticRegression):
        prediction_results=calculate_accuracy(trained_model,
            predictions_with_probabilities,
            X_test,
            lr_model_data['y_test'].tolist(),
            N=3)
    labels = unique_labels(
        lr_model_data['y_test'],
        y_pred
        )
    report = classification_report(lr_model_data['y_test'].tolist(),
        y_pred,
        labels=labels,
        target_names=[str(label) for label in labels],
        output_dict=True)
    return ({"report": report,
            "prediction_results": prediction_results
        })
    
def run_full_model_generation(smoke_test_N=None, 
    pca_feature_reduction=False,
    notes="Logistic regression for topic classification",
    upload_raw_data=False,
    model=LogisticRegression(max_iter=1000),
    transformed_embeddings=None):
    if isinstance(model, LogisticRegression):
                model_type="Logistic Regression"
    elif isinstance(model, XGBClassifier):
                model_type="XGB"
    elif isinstance(model, DecisionTreeClassifier):
                model_type="Decision Tree"
    elif isinstance(model, RandomForestClassifier):
                model_type="Random Forest"
    elif isinstance(model, AdaBoostClassifier):
                model_type="Ada Boost"
    elif isinstance(model, GradientBoostingClassifier):
                model_type="Gradient Boost"
    else:
        model_type=""
    if transformed_embeddings is None:
        tracemalloc.start()
        print("Read in data...")
        all_raw_data=read_dataset_from_file(raw_data_filename)
        print(f"Preprocess data for {model_type}...")
        df=data_preprocessing(all_raw_data, smoke_test_N)
        df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories', 'Topic']]
        df = df.drop_duplicates()
        start = time.perf_counter()
        print(f"Generate embeddings for {len(df)} items for {model_type}. Feature reduction: {pca_feature_reduction}")
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
    print(f"Create {model_type} data for {len(transformed_embeddings)} items...")    
    lr_model_data=create_model_data(transformed_embeddings, pca_feature_reduction, model=model)
    #lr_model_data=create_model_data(reduced_embeddings)
    start = time.perf_counter()
    print(f"Train model for {len(transformed_embeddings)} items. Feature reduction: {pca_feature_reduction}")
    trained_model=train_model(lr_model_data, prediction_model=model)
    save_versioned_pickle_file(trained_model, f'{model_type} Model', folder='./projects/am1_project/model')
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
        model_training_time=model_training_time,
        pca_feature_reduction=pca_feature_reduction,
        model_notes=notes
        ))
    print("Test model...")    
    test_results=test_model(trained_model, lr_model_data)
    model_name_version=f"{model_type} for topic classification"
    notes=notes
    input_example=convert_df_to_ndarray(lr_model_data['X_train'][:5])
    print(input_example)
    print("Register model on MLFlow...")
    print(notes)
    register_model_and_metrics(trained_model,
        model_type,
        model_name_version,
        test_results["report"],
        notes,
        input_example,
        raw_data_filename,
        prediction_results=test_results["prediction_results"],
        )

def run_full_model_generation_with_cross_validation(smoke_test_N=None, 
    pca_feature_reduction=False,
    notes="Logistic regression for topic classification",
    upload_raw_data=False,
    model=LogisticRegression(max_iter=1000)):
    tracemalloc.start()
    print("Read in data...")
    all_raw_data=read_dataset_from_file(raw_data_filename)
    print("Preprocess data...")
    df=data_preprocessing(all_raw_data, smoke_test_N)
    df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories', 'Topic']]
    start = time.perf_counter()
    print(f"Generate embeddings for {smoke_test_N} items...")
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
    skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
    )
    y=transformed_embeddings['topic']
    X=transformed_embeddings.drop('topic', axis=1)
    scoring_object={
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted"
        }
    if isinstance(model, LogisticRegression):
        all_classes = np.unique(y)
        top5_scorer = make_scorer(
            top_k_accuracy_score,
            k=5,
            labels=all_classes,
            response_method="predict_proba"
        )
        scoring_object["top_5_accuracy"] = top5_scorer
    if isinstance(model, XGBClassifier):
        le = LabelEncoder()
        y = le.fit_transform(y)
    scores=cross_validate(
        model,
        convert_df_to_ndarray(X),
        y,
        cv=skf,
        scoring=scoring_object
        )
    start = time.perf_counter()
    print(f"Train model for {smoke_test_N} items...")
    trained_model=LogisticRegression(max_iter=1000)
    trained_model.fit(convert_df_to_ndarray(X), y)
    model_training_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(StructuredMessage(message='Train model',
        application="am1",
        operation_type="train_am1_model",
        number_of_training_records=len(X),
        size_of_labels=X['summary_embeddings'].memory_usage(deep=True),
        size_of_categories=X['category_embeddings'].memory_usage(deep=True),
        current_memory_usage=current,
        peak_memory_usage=peak,
        model_training_time=model_training_time,
        pca_feature_reduction=pca_feature_reduction,
        model_notes=notes
        ))
    model_name_version=f"logistic_regression_for_topic_classification"
    register_model_and_cross_validation_metrics(trained_model,
        LogisticRegression,
        model_name_version,
        scores,
        notes,
        X[:5],
        raw_data_filename)

def generate_all_embeddings():
        tracemalloc.start()
        print("Read in data...")
        all_raw_data=read_dataset_from_file(raw_data_filename)
        print("Preprocess data...")
        df=data_preprocessing(all_raw_data)
        df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories', 'Topic']]
        start = time.perf_counter()
        print(f"Generate embeddings for all {len(df)} items...")
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
        save_versioned_pickle_file(transformed_embeddings, 'transformed_embeddings_from_pipeline', folder='./projects/am1_project/data')


    
    