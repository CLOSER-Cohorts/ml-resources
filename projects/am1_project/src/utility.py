import numpy as np

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

def get_items_with_no_labels(colectica_client):
    items_with_no_labels = []
    all_questions_variables = colectica_client.search_items(
            [colectica_client.item_code('Question'), colectica_client.item_code('Variable')],
            ReturnIdentifiersOnly=True,
            SearchLatestVersion=True,
            MaxResults=0)['Results']
    count = 0        
    for item in all_questions_variables:
        count = count + 1
        group_type=[]
        if item['ItemType']==colectica_client.item_code('Question'):
            group_type=colectica_client.item_code('Question Group')
        elif item['ItemType']==colectica_client.item_code('Variable'):
            group_type=colectica_client.item_code('Variable Group')
        topic=colectica_client.search_relationship_byobject(item['AgencyId'],
                item['Identifier'],
                item_types=group_type,
                Version=item['Version'],
                Descriptions=False)
        if len(topic)==0:
            items_with_no_labels.append(item)
        print(f"Found {len(items_with_no_labels)} items with no topics in {count} items")
    return items_with_no_labels

def convert_df_to_ndarray(df_data):
    input_feature_list=[]
    for input_feature in ['summary_embeddings',
        'category_embeddings',
        'item_type',
        'agency_id',
        'has_categories']:
        input_feature_list.append(np.vstack(df_data[input_feature]))
        X_test = np.hstack(
            input_feature_list
        )
    return X_test

