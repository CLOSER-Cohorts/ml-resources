import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from src.ml_resources import (
    calculate_accuracy,
    read_dataset_from_file
    )
from src.dataframe_utility import convert_df_to_ndarray

def get_urn_from_item(item):
   return f"urn:ddi:{item['AgencyId']}:{item['Identifier']}:{str(item['Version'])}"

def get_item_from_topic_name(topic_name,
    topic_type,
    containing_item,
    C,
    dataset_name="",
    groupsInDatasets=[],
    datasetToZeroGroupMappings={}):
    """Method for getting a topic item given the topic's name as a string (e.g. '11609'), the topic 
    type (e.g. Question Group, Variable Group), and the item within which that topic is contained 
    (e.g. a Physical Instance/Data File or a Data Collection object).

    Note that the topic_type input argument must be provided as a UUID (as specified at
    https://docs.colectica.com/repository/technical/item-type-identifiers/). Item types can be 
    mapped to their identifiers using the C.item_code function, e.g. C.item_code("Question Group"),
    C.item_code("Data Collection").

    Arguments:
        topic_name (str): the name of the topic we are searching for (e.g. '11609').
        topic_type (str): the type of the topic we are searching for.
        containing_item (dict): A dictionary containing details of the item containing the item being 
            reassigned.
        C (ColecticaObject): an authenticated ColecticaObject instance.

    Keyword arguments:
        groupsInDatasets (list): A list of dict objects that map the datasets to topic groups they contain.
        datasetToZeroGroupMappings (dict): A dictionary mapping dataset names to level zero topic groups. 

    Returns:
        list: A list containing Variable Groups/Question Groups items that represent topics.
    """
    item=[x for x in groupsInDatasets if x['DatasetName']==dataset_name 
        and x['VariableGroupName']==str(topic_name) and x['TopicType']==topic_type]
    if len(item)==1:
        if item[0]['VariableGroupUrn']=="NA":
            topic_groups=[]
        else:
            topic_groups=[C.get_item_json(
                item[0]['VariableGroupUrn'].split(":")[2],
                item[0]['VariableGroupUrn'].split(":")[3],
                version=item[0]['VariableGroupUrn'].split(":")[4]
        )]
    else:
        topic_groups = C.search_items(topic_type,
                     SearchSets=containing_item,
                     SearchTerms=[str(topic_name)],
                     SearchTargets="Name",
                     UsePrefixSearch=False)['Results']
        containing_level_zero_group = C.search_relationship_bysubject(containing_item['AgencyId'],
                containing_item['Identifier'],
                item_types=C.item_code('Variable Group'),
                Version=containing_item['Version'],
                Descriptions=True)
        if len(containing_level_zero_group)==1:
                containing_level_zero_group_item=C.get_item_json(containing_level_zero_group[0]['AgencyId'],
                containing_level_zero_group[0]['Identifier'],
                version=containing_level_zero_group[0]['Version'])
                if containing_level_zero_group_item['Concept'] == None:
                    datasetToZeroGroupMappings[get_urn_from_item(containing_item)]=[{
                        "AgencyId": containing_level_zero_group[0]['AgencyId'],
                        "Identifier": containing_level_zero_group[0]['Identifier'],
                        "Version": containing_level_zero_group[0]['Version'],
                        }]
        if len(topic_groups)==0:
            if get_urn_from_item(containing_item) not in datasetToZeroGroupMappings.keys():
                # If we cannot determine the level zero group for the dataset (i.e. topic_group is
                # empty) we must try to determine the level zero group by inspecting variables in 
                # the dataset...
                print((f"Cannot determine level zero group for dataset {get_urn_from_item(containing_item)}, " 
                    "inspecting variables..."))
                datasetVars=C.query_set(containing_item['AgencyId'], 
                    containing_item['Identifier'],item_types=[C.item_code('Variable')])
                level_zero_groups=[]
                count=0
                print(f"Verifying the level zero group for {len(datasetVars)} variables in dataset {get_urn_from_item(containing_item)}...")
                # First we try to find a group that is referenced by the containing item (e.g. a dataset).
                #level_zero_groups.extend(containing_level_zero_group)
                # If we find a group, that's the level zero group. Sometimes the reference to the level
                # zero group is missing from the containing item, so we will have to determine the level
                # zero group using the variables in the dataset.
                # We only determine the level zero group for a small sample of variables, for a faster runtime...
                if len(level_zero_groups)==0:
                    for var in datasetVars[0:4]:
                        varGroups=C.search_relationship_byobject(var['Item1']['Item3'], var['Item1']['Item1'], 
                            Version=var['Item1']['Item2'], item_types=[topic_type]) 
                        for varGroup in varGroups:
                            count=count+1
                            var_group_item=C.get_item_json(varGroup['Item1']['Item3'],
                                varGroup['Item1']['Item1'], 
                                version=varGroup['Item1']['Item2'])
                            level_zero_group=get_level_zero_group_for_topic(var_group_item, C)
                            if level_zero_group is not None:
                                level_zero_groups.append(level_zero_group)            
                    containing_level_zero_group = []
                # If all the level zero groups we have found are the same group, we can assume that this is 
                # the level zero group for the dataset specified in the containing_item argument...
                if len(set([x[0][2].text for x in level_zero_groups]))==1:
                    containing_level_zero_group = [{
                    "AgencyId": level_zero_groups[0][0][1].text,
                    "Identifier": level_zero_groups[0][0][2].text,
                    "Version": level_zero_groups[0][0][3].text,
                    }]
                    datasetToZeroGroupMappings[get_urn_from_item(containing_item)]=containing_level_zero_group
                else:
                    containing_level_zero_group = []
                # Do a search for the first three numbers of the topic group, and then filter
                # the results in a list comprehension to find the exact match, because it's
                # quicker than just searching for the exact match directly.
                topic_groups = [x for x in C.search_items(topic_type,
                     SearchSets=containing_level_zero_group,
                     SearchTerms=[str(topic_name)[0:3]],
                     UsePrefixSearch=True, # returns results if they begin with the value in SearchTerms
                     SearchTargets="Name")['Results'] if x['ItemName']['en-GB']==str(topic_name)]
            else:
                containing_level_zero_group=datasetToZeroGroupMappings[get_urn_from_item(containing_item)]
                # Do a search for the first three numbers of the topic group, and then filter
                # the results in a list comprehension to find the exact match, because it's
                # quicker than just searching for the exact match directly.
                topic_groups = [x for x in C.search_items(topic_type,
                     SearchSets=containing_level_zero_group,
                     SearchTerms=[str(topic_name)[0:3]],
                     UsePrefixSearch=True,  # returns results if they begin with the value in SearchTerms
                     SearchTargets="Name")['Results'] if x['ItemName']['en-GB']==str(topic_name)]
        """
        else:
            if len(containing_level_zero_group)==1:
                containing_level_zero_group_item=C.get_item_json(containing_level_zero_group[0]['AgencyId'],
                containing_level_zero_group[0]['Identifier'],
                version=containing_level_zero_group[0]['Version'])
                if containing_level_zero_group_item['Concept'] == None:
                    datasetToZeroGroupMappings[get_urn_from_item(containing_item)]=[{
                        "AgencyId": containing_level_zero_group[0]['AgencyId'],
                        "Identifier": containing_level_zero_group[0]['Identifier'],
                        "Version": containing_level_zero_group[0]['Version'],
                        }]
        """
    for topic_group in topic_groups:
        if topic_group['ItemName']['en-GB']==str(topic_name) and len(item)==0:
                groupsInDatasets.append({
                    "DatasetName": dataset_name,
                    "VariableGroupName": str(topic_name),
                    "VariableGroupUrn": "urn:ddi:" + topic_group['AgencyId'] + ":" + topic_group['Identifier'] + ":" + str(topic_group['Version']),
                    "TopicType": topic_type
                })
    if len(item)==0:
        groupsInDatasets.append({
                    "DatasetName": dataset_name,
                    "VariableGroupName": str(topic_name),
                    "VariableGroupUrn": "NA",
                    "TopicType": topic_type
                })
    return [x for x in topic_groups if x['ItemName']['en-GB']==str(topic_name)]

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

def test_model(trained_model,
    model_data, 
    feature_columns,
    all_trained_models_cross_val_confidence=None,
    test_field_X='X_test',
    test_field_y='y_test',
    agency_id=None):
    #input_feature_list=[]
    X_test=convert_df_to_ndarray(model_data[test_field_X], input_features=feature_columns)
    if agency_id not in all_trained_models_cross_val_confidence.keys():
        all_trained_models_cross_val_confidence[agency_id]={}   
    """
    for input_feature in ['summary_embeddings', 
        'category_embeddings', 'item_type', 'agency_id', 'has_categories']:
        input_feature_list.append(np.vstack(model_data['X_test'][input_feature]))
        X_test = np.hstack(
        input_feature_list
        )
    """
    y_pred=trained_model.predict(X_test)
    predictions_with_probabilities=trained_model.predict_proba(X_test)
    max_values = np.max(predictions_with_probabilities, axis=0)
    print(max_values)
    print(y_pred)
    for pred, max_value in zip(y_pred, max_values):
        if pred not in all_trained_models_cross_val_confidence[agency_id].keys():
          all_trained_models_cross_val_confidence[agency_id][pred]=[]
        all_trained_models_cross_val_confidence[agency_id][pred].append(
           max_value
        )
    prediction_results = None
    if isinstance(trained_model, XGBClassifier):
        le = LabelEncoder()
        y_test = le.fit_transform(model_data[test_field_y])
    else:
        y_test=model_data[test_field_y]
    labels = unique_labels(
        y_test,
        y_pred
        )
    prediction_results=calculate_accuracy(trained_model,
            predictions_with_probabilities,
            X_test,
            #model_data['y_test'].tolist(),
            y_test.tolist(),
            N=5)
    if not isinstance(trained_model, XGBClassifier) and not isinstance(trained_model, MLPClassifier):
        roc_auc_result=roc_auc_score(y_test, 
            predictions_with_probabilities, 
            multi_class='ovr', 
            labels=trained_model.classes_)
    else:
        roc_auc_result = None
    report = classification_report(y_test.tolist(),
        y_pred,
        labels=labels,
        target_names=[str(label) for label in labels],
        output_dict=True)
    return ({"report": report,
            "prediction_results": prediction_results,
            "roc_auc_score": roc_auc_result
        })


def test_model_all_studies(trained_model,
    model_data, 
    feature_columns,
    all_trained_models_cross_val_confidence=None,
    test_field_X='X_test',
    test_field_y='y_test',
    agency_id=None):
    """Version of test_model for model which is trained on all studies
    """
    #input_feature_list=[]
    X_test=convert_df_to_ndarray(model_data[test_field_X], input_features=feature_columns)
    """
    for input_feature in ['summary_embeddings', 
        'category_embeddings', 'item_type', 'agency_id', 'has_categories']:
        input_feature_list.append(np.vstack(model_data['X_test'][input_feature]))
        X_test = np.hstack(
        input_feature_list
        )
    """
    y_pred=trained_model.predict(X_test)
    predictions_with_probabilities=trained_model.predict_proba(X_test)
    max_values = np.max(predictions_with_probabilities, axis=0)
    print(max_values)
    print(y_pred)
    for pred, max_value in zip(y_pred, max_values):
        if pred not in all_trained_models_cross_val_confidence.keys():
          all_trained_models_cross_val_confidence[pred]=[]
        all_trained_models_cross_val_confidence[pred].append(
           max_value
        )
    prediction_results = None
    if isinstance(trained_model, XGBClassifier):
        le = LabelEncoder()
        y_test = le.fit_transform(model_data[test_field_y])
    else:
        y_test=model_data[test_field_y]
    labels = unique_labels(
        y_test,
        y_pred
        )
    prediction_results=calculate_accuracy(trained_model,
            predictions_with_probabilities,
            X_test,
            #model_data['y_test'].tolist(),
            y_test.tolist(),
            N=5)
    if not isinstance(trained_model, XGBClassifier) and not isinstance(trained_model, MLPClassifier):
        roc_auc_result=roc_auc_score(y_test, 
            predictions_with_probabilities, 
            multi_class='ovr', 
            labels=trained_model.classes_)
    else:
        roc_auc_result = None
    report = classification_report(y_test.tolist(),
        y_pred,
        labels=labels,
        target_names=[str(label) for label in labels],
        output_dict=True)
    return ({"report": report,
            "prediction_results": prediction_results,
            "roc_auc_score": roc_auc_result
        })


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
            validation_data=pd.concat([validation_data, 
                test_data[test_data['ContainedIn']==x]])
            validation_data_y= all_data['y_test'].loc[validation_data.index]
            len_validation = len_validation + len(test_data[test_data['ContainedIn']==x])
        else:
            reduced_test_data=pd.concat([reduced_test_data,
                test_data[test_data['ContainedIn']==x]])
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

def create_embeddings(model_all_studies=True, 
    agencies=['uk.iser.ukhls', 'uk.whitehall2', 'uk.cls.nextsteps', 'uk.lha', 'uk.wchads', 'uk.cls.bcs70', 'uk.alspac', 'uk.mrcleu-uos.sws', 'uk.genscot', 'uk.mrcleu-uos.hcs', 'uk.mrcleu-uos.heaf']):
    all_test_results={}
    all_trained_models={}
    if model_all_studies:
        all_embeddings_X_train=pd.DataFrame()
        all_embeddings_y_train=pd.DataFrame()
        all_embeddings_X_test=pd.DataFrame()
        all_embeddings_y_test=pd.DataFrame()    
    for agency in agencies:
        print(agency)
        if not model_all_studies:
            all_embeddings_X_train=pd.DataFrame()
            all_embeddings_y_train=pd.DataFrame()
            all_embeddings_X_test=pd.DataFrame()
            all_embeddings_y_test=pd.DataFrame()
        # I'm reading from a file where the data is already in a dict object with the keys
        # >>> embeddings.keys()
        # dict_keys(['X_train', 'X_test', 'y_train', 'y_test'])
        embeddings=read_dataset_from_file(f'./projects/am1_project/data/unfiltered_embeddings/{agency}_model_embeddings/{agency}_model_embeddings_2.pickle')
        all_embeddings_X_train=pd.concat([all_embeddings_X_train, embeddings['X_train']], ignore_index=True)
        all_embeddings_X_test=pd.concat([all_embeddings_X_test, embeddings['X_test']], ignore_index=True)
        all_embeddings_y_train=pd.concat([all_embeddings_y_train, embeddings['y_train']], ignore_index=True)
        all_embeddings_y_test=pd.concat([all_embeddings_y_test, embeddings['y_test']], ignore_index=True)   
        embeddings['X_train'] = all_embeddings_X_train
        embeddings['X_test'] = all_embeddings_X_test
        embeddings['y_train'] = all_embeddings_y_train
        embeddings['y_test'] = all_embeddings_y_test
        df_training_embeddings=embeddings['X_train']
        df_training_embeddings['Topic']=embeddings['y_train']
        df_training_embeddings=df_training_embeddings[~df_training_embeddings["Topic"].str.startswith("116", na=False)]
        df_test_embeddings=embeddings['X_test']
        df_test_embeddings['Topic']=embeddings['y_test']
        df_test_embeddings=df_test_embeddings[~df_test_embeddings["Topic"].str.startswith("116", na=False)]
        df_test_embeddings["TextLabel"] = df_test_embeddings["TextLabel"].str.replace("\xa0", " ", regex=False)
        df_test_embeddings=remove_duplicate_textlabels(df_training_embeddings, df_test_embeddings, col='TextLabel')
        df_test_embeddings=filter_by_topics(df_test_embeddings, df_training_embeddings, col="Topic")
        df_training_embeddings=filter_by_topics(df_training_embeddings, df_test_embeddings, col="Topic")
        embeddings['X_train']=df_training_embeddings.drop('Topic', axis=1)
        embeddings['y_train']=df_training_embeddings['Topic']
        embeddings['X_test']=df_test_embeddings.drop('Topic', axis=1)
        embeddings['y_test']=df_test_embeddings['Topic']
        embeddings['X_train'] = embeddings['X_train'].reset_index(drop=True)
        embeddings['y_train'] = embeddings['y_train'].reset_index(drop=True)
        embeddings['X_test'] = embeddings['X_test'].reset_index(drop=True)
        embeddings['y_test'] = embeddings['y_test'].reset_index(drop=True)
    return embeddings
