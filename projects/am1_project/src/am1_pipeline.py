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
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import top_k_accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer, util
llm_model = SentenceTransformer('all-MiniLM-L6-v2')
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file,
    filter_values_by_length,
    apply_pipeline,
    convert_dictionary_to_dataframe,
    create_model_data_object,
    train_model
    )
from projects.am1_project.src.am1_mlflow import (
    register_model_and_metrics,
    register_model_and_cross_validation_metrics
    )
from projects.am1_project.src.utility import convert_df_to_ndarray    
from src.logging.utility import StructuredMessage, setup_logging
from utility import test_model

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py.
# We assume the code is being run from the repository root directory (ml-resources).

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")
raw_data_filename = './projects/am1_project/data/all_raw_data/all_raw_data_1.pickle'

# We need to filter that JSON dict because multiple categories in a dataframe row
# are too messy to deal with
def filter_items(raw_data):
    filtered_items=filter_values_by_length(raw_data, "TextLabel", 10)
    filtered_items_by_number_of_categories=filter_values_by_length(filtered_items,
        "ItemCategories", 3, include_zero_length_items=True)
    filtered_items=filtered_items_by_number_of_categories
    return filtered_items


def encode_columns(df, columns):
    out = df.copy()
    for col in columns:
        embeddings = llm_model.encode(df[col].tolist(), show_progress_bar=True)
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])],
            index=df.index
        )
        out = pd.concat([out, emb_df], axis=1)
    return out

def encode_columns_narrow(df, columns):
    out = df.copy()
    for col in columns:
        embeddings = llm_model.encode([str(x) for x  in df[col].tolist()], show_progress_bar=True)
        out[f'{col}_embeddings'] = list(embeddings)
    return out

def remove_single_instances(df, topic_column="Topic"):
    class_proportions = df[topic_column].value_counts()
    items_with_unique_topics=list(class_proportions[class_proportions<2].index)
    df=df[~df[topic_column].isin(items_with_unique_topics)].reset_index(drop=True)
    return df

def create_model_data(data_for_model, model=None, smoke_test_N=None):
    if smoke_test_N is not None:
        data_for_model = data_for_model.iloc[:smoke_test_N]
    y=data_for_model['Topic']
    X=data_for_model.drop('Topic', axis=1)
    
    if isinstance(model, XGBClassifier):
        le = LabelEncoder()
        y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 20% test set
        random_state=42,     # for reproducibility
        stratify=y           # ensures balanced class proportions
    )
    model_data=create_model_data_object(X_train, X_test, y_train, y_test)
    return model_data

def create_model_data_for_new_wave_model(training_data_for_model, 
    test_data_for_model,
    model=None, smoke_test_N=None):
    if smoke_test_N is not None:
        data_for_model = data_for_model.iloc[:smoke_test_N]
    y_train=training_data_for_model['Topic']
    X_train=training_data_for_model.drop('Topic', axis=1)
    y_test=test_data_for_model['Topic']
    X_test=test_data_for_model.drop('Topic', axis=1)
    
    
    if isinstance(model, XGBClassifier):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)  
    
    model_data=create_model_data_object(X_train, X_test, y_train, y_test)
    return model_data


def preprocess_data_frame(df, 
    item_categories_col='ItemCategories',
    has_categories_col='HasCategories', 
    agency_col='AgencyId',
    item_type_col='ItemType', 
    topic_col='Topic',
    contained_in_col='ContainedIn'):
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
    df[contained_in_col] = pd.Categorical(df[contained_in_col]).codes
    # Now we perform some data cleaning. Resetting the index is important so the pipeline
    # operations will work (the indices have to be continuous numeric values with no gaps)
    df = df.dropna(subset=[topic_col])
    # Save the ids in our data for future use when deciding if there is data available
    # on the Colectica repository that isn't in our training/testing data
    df=df.reset_index(drop=True)
    return df


def data_preprocessing(items_for_dataframe, smoke_test_N=None, topic_column="Topic", filter=True):
    # We now run quality control code to filter items which don't meet certain criteria. 
    # E.g. the question summary is too short, the question has fewer than N categories
    # associated with it, the question summary contains text, a question has a set of
    # categories associated with it that are not deemed to have predictive value, e.g. yes/no
    tracemalloc.start()
    start = time.perf_counter()
    print(f"Filtering items: {filter}")
    if filter:
        items_for_dataframe=filter_items(items_for_dataframe)
    # Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc
    df=convert_dictionary_to_dataframe(items_for_dataframe)
    df=preprocess_data_frame(df)
    if smoke_test_N is not None:
        df=df[0:smoke_test_N]
    print(df)
    df = df[['TextLabel', 'ItemCategories', 'ItemType', 'AgencyId', 'HasCategories', 'ContainedIn', 'Topic']]
    df = df.drop_duplicates()
    df = df[df["ItemType"].isin([0, 1])]
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

def generate_embeddings(raw_model_data, 
    model=LogisticRegression(max_iter=1000), 
    smoke_test_N=None, 
    embeddings_file_name="",
    embeddings_folder_name="",
    filter=True):
        tracemalloc.start()
        print(f"Preprocess data...")
        if set(raw_model_data.keys()) == {'test', 'training'}:
            print("Seperate objects for train and test data found.")
            df_training=data_preprocessing(raw_model_data['training'],
                smoke_test_N=smoke_test_N,
                filter=filter)
            df_test=data_preprocessing(raw_model_data['test'], smoke_test_N=smoke_test_N, filter=filter)
            model_data=create_model_data_for_new_wave_model(df_training, 
                df_test,
                model=model,
                smoke_test_N=smoke_test_N)
        else:
            print("No seperate objects for train and test data defined, train_test_split will be executed.") 
            df=data_preprocessing(raw_model_data, smoke_test_N)
            model_data=create_model_data(df, model=model)
        tracemalloc.reset_peak()
        start = time.perf_counter()
        print(f"Generate embeddings for {len(model_data['X_train']) + len(model_data['X_test'])} items.")
        current, peak = tracemalloc.get_traced_memory()
        model_data['X_train'] = encode_columns_narrow(model_data['X_train'], ['TextLabel', 'ItemCategories'])
        model_data['X_test'] = encode_columns_narrow(model_data['X_test'], ['TextLabel', 'ItemCategories'])
        save_versioned_pickle_file(model_data, f'{embeddings_file_name}_model_embeddings', folder=f'./projects/am1_project/data/{embeddings_folder_name}')
        embeddings_generation_time = time.perf_counter() - start
        tracemalloc.reset_peak()
        logger.info(StructuredMessage(message='Generate embeddings',
            application="am1",
            operation_type="generate_embeddings",
            number_of_embeddings=len(model_data['X_train']) + len(model_data['X_test']),
            size_of_training_labels=model_data['X_train']['TextLabel_embeddings'].memory_usage(deep=True),
            size_of_test_labels=model_data['X_test']['TextLabel_embeddings'].memory_usage(deep=True),
            size_of_training_categories=model_data['X_train']['ItemCategories_embeddings'].memory_usage(deep=True),
            size_of_test_categories=model_data['X_test']['ItemCategories_embeddings'].memory_usage(deep=True),
            current_memory_usage=current,
            peak_memory_usage=peak,
            embeddings_generation_time=embeddings_generation_time
            ))
        return model_data
    

def run_full_model_generation(smoke_test_N=None, 
    pca_feature_reduction=False,
    notes="Logistic regression for topic classification",
    upload_raw_data=False,
    model=LogisticRegression(max_iter=1000),
    model_data=None,
    model_name="",
    feature_columns=['ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
):
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
        model_type="MLPClassifier"
    if smoke_test_N is not None:
        model_data['X_train'] = model_data['X_train'][:smoke_test_N]
        model_data['y_train'] = model_data['y_train'][:smoke_test_N]
        model_data['X_test'] = model_data['X_test'][:(int(smoke_test_N/3))]
        model_data['y_test'] = model_data['y_test'][:(int(smoke_test_N/3))]        
    if pca_feature_reduction:
        pca = PCA(n_components=128)
        model_data['X_train']['TextLabel_embeddings'] = list(pca.fit_transform(
                convert_df_to_ndarray(model_data['X_train'],
                    input_features=['TextLabel_embeddings'])
            )) 
        model_data['X_train']['ItemCategories_embeddings'] = list(pca.fit_transform(
                convert_df_to_ndarray(model_data['X_train'],
                    input_features=['ItemCategories_embeddings'])
            ))
        model_data['X_test']['TextLabel_embeddings'] = list(pca.transform(
                convert_df_to_ndarray(model_data['X_test'],
                    input_features=['TextLabel_embeddings'])
            )) 
        model_data['X_test']['ItemCategories_embeddings'] = list(pca.transform(
                convert_df_to_ndarray(model_data['X_test'],
                    input_features=['ItemCategories_embeddings'])
            ))
    #tracemalloc.start()
    tracemalloc.stop()
    start = time.perf_counter()
    print(f"Train {model_type} model for {len(model_data['X_train'])} items. Feature reduction: {pca_feature_reduction}")
    #feature_columns=['ItemType', 'HasCategories', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    print(f"feature columns: {feature_columns}")
    trained_model=train_model(model_data, prediction_model=model, feature_columns=feature_columns)
    save_versioned_pickle_file(trained_model, f'{model_type} Model', folder='./projects/am1_project/model')
    model_training_time = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    logger.info(StructuredMessage(message='Train model',
        application="am1",
        operation_type="train_am1_model",
        number_of_training_records=len(model_data['X_train']),
        size_of_training_labels=model_data['X_train']['TextLabel_embeddings'].memory_usage(deep=True),
        size_of_test_labels=model_data['X_test']['TextLabel_embeddings'].memory_usage(deep=True),
        size_of_training_categories=model_data['X_train']['ItemCategories_embeddings'].memory_usage(deep=True),
        size_of_test_categories=model_data['X_test']['ItemCategories_embeddings'].memory_usage(deep=True),
        current_memory_usage=current,
        peak_memory_usage=peak,
        model_training_time=model_training_time,
        pca_feature_reduction=pca_feature_reduction,
        model_notes=notes
        ))
    #tracemalloc.stop()
    print("Test model...")    
    test_results=test_model(trained_model, model_data, feature_columns)
    model_name_version=f"{model_type} for topic classification"
    notes=notes
    input_example=model_data['X_train'][:5]
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
        roc_auc_score=test_results["roc_auc_score"]
        )     