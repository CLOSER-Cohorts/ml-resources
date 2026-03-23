from colectica_api import ColecticaObject
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ml_resources.data import colectica_utility
from ml_resources import read_dataset_from_file, save_versioned_pickle_file
colectica_client = colectica_utility.C
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import combinations
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

all_relationships={}
all_unique_relationships={}
all_principal_components={}
item_types = [colectica_client.item_code('Series'),
colectica_client.item_code('Study'),
colectica_client.item_code('Data File'),
colectica_client.item_code('Organization'),
colectica_client.item_code('Instrument'),
colectica_client.item_code('Data Collection')
]

sweeps_info=colectica_utility.get_sweeps()

# Here is sample code for printing all the names of sweeps in each study...

for x in sweeps_info.keys():
    print(x)
    for y in sweeps_info[x]['SweepNames']:
        print(y)
    print("\n")

# Save the sweeps to a file...

save_versioned_pickle_file(sweeps_info, 'sweeps_info', folder='../projects/am2_project/data')

# ...and here is some sample code for getting a specific study from the sweep_info object.
len([x for x in sweeps_info['uk.cls.ncds']['SweepItems'] if x['ItemName']['en-GB']=='Age 44 Biomedical Survey (2002)'] )

item_types_string=['Action', 'Archive', 'Category', 'Category Group', 'Category Set', 'ClassificationCorrespondenceTable', 'ClassificationFamily', 'ClassificationIndex', 'ClassificationItem', 'ClassificationLevel', 'ClassificationSeries', 'Code List Group', 'Code List Set', 'Code Set', 'Concept', 'Concept Group', 'Concept Set', 'Conceptual Component', 'Conceptual Variable', 'Conceptual Variable Group', 'Conceptual Variable Set', 'Conditional', 'Control Construct Group', 'Control Construct Set', 'Data Collection', 'Data File', 'Data Layout', 'DataCollection Methodology', 'General Instruction', 'Generation Instruction', 'Individual', 'Instruction Group', 'Instrument', 'Instrument Group', 'Instrument Set', 'Interviewer Instruction', 'Interviewer Instruction Set', 'Logical Product', 'Loop', 'Managed Representation Group', 'Managed Representation Set', 'MeasurementItem', 'MeasurementConstruct', 'Metadata Package', 'NCube', 'NCube Group', 'NCube Set', 'Organization', 'Organization Group', 'Organization Set', 'OtherMaterial', 'OtherMaterialGroup', 'OtherMaterialScheme', 'Physical Data Product', 'Physical Structure', 'PhysicalStructure Set', 'Processing Event', 'Processing Event Group', 'Processing Event Set', 'Processing Instruction Group', 'Processing Instruction Scheme', 'Project', 'Quality Standard', 'Quality Statement', 'Quality Statement Group', 'Quality Statement Set', 'Question', 'Question Activity', 'Question Block', 'Question Grid', 'Question Group', 'Question Set', 'RecordLayout', 'RecordLayout Set', 'Repeat', 'Represented Variable', 'Represented Variable Group', 'Represented Variable Set', 'Reusable Missing Value', 'Sequence', 'Series', 'Statement', 'StatisticalClassification', 'Study', 'SubSeries', 'UnitType', 'UnitTypeScheme', 'UnitTypeGroup', 'Universe', 'Universe Group', 'Universe Set', 'Variable', 'Variable Group', 'Variable Set', 'Variable Statistic', 'While']
item_types_string=['Data File', 'Data Collection', 'Variable Statistic', 'While', 'Instrument']
#START WITH ONE DATA TYPE, E.G. DATA FILE, BUILD IN THAT
for item_type in [colectica_client.item_code(x) for x in item_types_string]:
  if item_type==colectica_client.item_code('Instrument'):
    items = colectica_client.search_items([item_type], MaxResults=1500, ReturnIdentifiersOnly=True, SearchLatestVersion=True)['Results']
    parents2=[]    
    df_relationships = pd.DataFrame()
    count=0
    print(f"{colectica_client.item_code_inv(item_type)}")
    for item in items:
      print(item)
      if item['Identifier'] != '4f1fa78e-ff60-4a85-bd3c-aace9da5955f':
        count=count+1
        print(f"{count} of {len(items)}")
        child=colectica_client.search_relationship_bysubject(item['AgencyId'], 
            item['Identifier'],
            Version=item['Version'])
        parent=colectica_client.search_relationship_byobject(item['AgencyId'],
            item['Identifier'],
            Version=item['Version'])
        print("Get descendants...")
        descendants=colectica_client.query_set(item['AgencyId'],
            item['Identifier'],
            version=item['Version'])
        print("Get ancestors...")
        ancestors=colectica_client.query_set(item['AgencyId'],
            item['Identifier'],
            version=item['Version'],
            reverseTraversal=True)
        descendantTypes=set([colectica_client.item_code_inv(x['Item2']) for x in descendants])
        ancestorTypes=set([colectica_client.item_code_inv(x['Item2']) for x in ancestors])
        newRow={}
        if  len(parent)>0:
            for x in list(descendantTypes) + list(ancestorTypes):
                if x in [colectica_client.item_code_inv(y['Item2']) for y in parent+child]:
                    newRow[x] = 5.0
                    parents2.append(parent[0])
                else:
                    newRow[x] = 1.0
        # Ensure all columns in new_row exist in df
        for key in newRow:
            if key not in df_relationships.columns:
                df_relationships[key] = 0 
        #df_relationships.loc[len(df_relationships)] = newRow
        df_relationships.loc[f"{item['AgencyId']}:{item['Identifier']}"] = newRow
        df_relationships = df_relationships.replace({np.nan: 0})
        df_relationships_unique = df_relationships.drop_duplicates()



#principal_components = generate_pca_data(X_train)

def find_clusters(pca_data):
    scores = []
    for k in range(2,min(10,len(pca_data))):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(pca_data)
            score = silhouette_score(pca_data, labels)
            scores.append(score)
    optimum_K = scores.index(max(scores))+2
    kmeans = KMeans(optimum_K)
    labels = kmeans.fit_predict(pca_data)
    return labels

def generate_data_for_classification(item_type, pca_data, all_data):
    labels=find_clusters(pca_data)
    unique_data_rows = all_data.drop_duplicates()
    final_dataset=pd.DataFrame({}, columns=['x', 'y', 'Distance', 'ItemType', 'Flagged'])
    print(f"THERE ARE {len(set(labels.tolist()))} CLUSTERS")
    for a in set(labels.tolist()):
                    indices_a = [i for i, x in enumerate(labels) if x == a]
                    matching_rows=pd.DataFrame({})
                    for index_a in indices_a:
                        matching_rows=pd.concat([matching_rows, all_data[all_data.eq(unique_data_rows.iloc[index_a]).all(axis=1)]])
                    print(f"CLUSTER {a}, {len(matching_rows)} ITEMS")
                    print(matching_rows)
                    print(matching_rows.index)
                    clusters_need_attention="a"
                    while clusters_need_attention not in ["y", "n"]:
                        clusters_need_attention = input("Does this cluster contain broken items? y/n\n")
                    if clusters_need_attention == 'y':
                        flagged=1
                    else:
                        flagged=0
                    indices_a = [i for i, x in enumerate(labels) if x == a]
                    for index_a in indices_a:
                        row_for_final_dataset = pca_data[index_a].tolist()
                        row_for_final_dataset.extend([math.dist(pca_data[index_a], [0,0]), item_type, flagged])
                        final_dataset.loc[len(final_dataset)] = row_for_final_dataset
    return final_dataset

def generate_data_for_test(pca_data, item_type):
    final_dataset=pd.DataFrame({}, columns=['x', 'y', 'Distance', 'ItemType'])
    for point in pca_data:
        row_for_final_dataset = point.tolist()
        row_for_final_dataset.extend([math.dist(pca_data[index_a], [0,0]), item_type])
        final_dataset.loc[len(final_dataset)] = row_for_final_dataset
    return final_dataset



def generate_pca_data(X_train, item_type, dataset_name="data"):
    #all_relationships[colectica_client.item_code_inv(item_type)] = X_train
    #all_unique_relationships[colectica_client.item_code_inv(item_type)] = unique_relationships_profile
    pca_data = PCA(n_components=2)
    #print(unique_relationships_profile.fillna(0))
    #unique_relationships_profile = X_train.drop_duplicates().fillna(0)
    if len(unique_relationships_profile)>1:
        principalComponents = pca_data.fit_transform(
            unique_relationships_profile)
        count=0
        plt.title(f"Plot for {item_type}")
        for i in principalComponents:
            plt.scatter(i[0], i[1])
            numberInCluster=len(X_train[(X_train==unique_relationships_profile.iloc[count,:]).all(axis=1)])
            plt.text(i[0], i[1], (str(count)+": "+str(numberInCluster)))
            count=count+1
        all_principal_components[item_type] = principalComponents
        plt.show(block=False)
        plt.savefig(f"outliers_{item_type}_{dataset_name}_2.png")
        plt.close()
        return principalComponents

item_types = [
'Data File',
'Instrument',
'Data Collection'
]

#item_type='Data Collection'

allData={}
for item_type in item_types:
    if input(f"Do you want to process the {item_type} items? ") in ['y', 'Y']:
        test_size_value=input("What proportion of the items do you want to put aside for testing? ")
        X_train, X_test = train_test_split(
            df_relationships,
            test_size=float(test_size_value)
        )
        # Generate training data, and train a decision tree
        input_for_pca=X_train.drop_duplicates().fillna(0)
        pca_data=generate_pca_data(X_train, item_type, "training")
        final_dataset=generate_data_for_classification(item_type, pca_data, X_train)
        dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        model_name_version=f"{item_type}_decision_tree_classifier_for_error_detection_v1"
        training_data=f"{len(X_train)}_{item_type}_items"
        #final_dataset["ItemType"] = final_dataset["ItemType"].astype("category").cat.codes
        X=pd.DataFrame(final_dataset[['ItemType', 'Distance']])
        X["ItemType"] = X["ItemType"].astype("category").cat.codes
        allData[item_type]=final_dataset
        y=final_dataset[['Flagged']]
        dtc.fit(X, y)
        # Generate test data, and calculate the accuracy of the decision tree
        print("Now we will run tests on the data we set aside...")
        input_for_pca=X_test.drop_duplicates().fillna(0)
        test_pca_data = generate_pca_data(input_for_pca, item_type, "test")
        test_dataset = generate_data_for_classification(item_type, test_pca_data, X_test)
        test_dataset.index = input_for_pca.index
        test_dataset["ItemType"] = test_dataset["ItemType"].astype("category").cat.codes
        #test_dataset=generate_data_for_test(test_pca_data, item_type, "test_data")
        y_pred=dtc.predict(pd.DataFrame(test_dataset[['ItemType', 'Distance']]))
        indices_flagged = [i for i, x in enumerate(y_pred) if x == 1]
        y_test=test_dataset['Flagged']
        y_test.values.tolist().extend(y_pred.tolist())
        target_values=[str(x) for x in set(y_test) | set(y_pred)]
        report = classification_report(y_test, y_pred, target_names=target_values)
        print(report)
        notes_on_experiment = input("Enter any notes on this experiment you wish to record: ")
        with open(f"classification_report_{item_type}.txt", "w") as f:
            _ = f.write(f"Classification report for {item_type}\n\n")
            _ = f.write(f"Notes: \n{notes_on_experiment}\n\n")
            _ = f.write(f"Model name and version: \n{model_name_version}\n\n")
            _ = f.write(f"Training data: \n{training_data}\n\n")
            _ = f.write(f"Classification report: \n{report}\n\n")
            _ = f.write("\nThese items have been flagged as needing attention.\n\n")
            for index_flagged in indices_flagged:
                _ = f.write(test_dataset.index[index_flagged])
        print(f"\n\n\nFinished {item_type}")

[str(x) for x in set(y_test)][str(x) for x in set(y_pred)]




"""
inertia = []
for k in range(1,10):
    model = KMeans(n_clusters=k)
    model.fit(principalComponents)
    inertia.append(model.inertia_)
"""

all_principal_components
scores=[]
for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(principalComponents)
    score = silhouette_score(principalComponents, labels)
    scores.append(score)

item_types_2 = ['Series',
'Study',
'Data File',
'Organization',
'Instrument',
'Data Collection'
]

all_training_data=pd.DataFrame({})


trainingData2=pd.DataFrame({}, columns=['x', 'y', 'Distance', 'ItemType', 'Flagged'])
for pc in all_principal_components.items():
#for pc in [all_principal_components['Data Collection']]:
 #  if pc[0]=='Series':
 if len(pc[1])>2 and pc[0] == 'Data File': # in item_types_2:
    print("\nCHECKING DATA TYPES: *******************" + pc[0] + "******************")
    scores = []
    for k in range(2,min(10,len(pc[1]))):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(pc[1])
            score = silhouette_score(pc[1], labels)
            scores.append(score)
    optimum_K = scores.index(max(scores))+2
    kmeans = KMeans(optimum_K)
    labels = kmeans.fit_predict(pc[1])
        #for x in range(0, optimum_K):
    print(f"THERE ARE {len(set(labels.tolist()))} CLUSTERS")
    #distances=[]
    #coordinates=[]
    #for x in set(labels.tolist()):
    #    distances.append(math.dist(all_principal_components[pc[0]][labels.tolist().index(x)], [0,0]))
    #    coordinates.append(all_principal_components[pc[0]][labels.tolist().index(x)].tolist())
    #print(distances)
    #for a, b in combinations(set(labels.tolist()), 2):
    X_train_clusters={}  
    for a in set(labels.tolist()):
                #biggest_differences=(abs(all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)]
                #    -all_unique_relationships[pc[0]].iloc[labels.tolist().index(b)])==5.0)
                #if len(biggest_differences.index[biggest_differences==True])>0:
                    #print(f"{a}, {b}")
                    distances=[]
                    coordinates=[]
                    distances.append(math.dist(all_principal_components[pc[0]][labels.tolist().index(a)], [0,0]))
                    coordinates=(all_principal_components[pc[0]][labels.tolist().index(a)].tolist())
                    print(f"{a}")
                    print(distances)
                    print(coordinates)
                    #print((f"DIFFERENCE: {all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)].name}, "
                    #    f"{all_unique_relationships[pc[0]].iloc[labels.tolist().index(b)].name}"))
                    indices_a = [i for i, x in enumerate(labels) if x == a]
                    #indices_b = [i for i, x in enumerate(labels) if x == b]
                    matching_rows = {}
                    matching_rows['a'] = all_relationships[pc[0]][all_relationships[pc[0]].eq(all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)]).all(axis=1)]
                    matching_rows['a']['ItemType'] = pc[0]
                    matching_rows['a']['Flagged'] = 0
                    matching_rows['a']['ItemTypesForInspection'] = ""
                    print(f"CLUSTER {a}, {len(matching_rows['a'])} ITEMS")
                    print(matching_rows['a'])
                    clusters_need_attention="a"
                    #print(biggest_differences.index[biggest_differences==True])
                    while clusters_need_attention not in ["y", "n"]:
                        clusters_need_attention = input("Does this cluster contain broken items? y/n\n")
                    if clusters_need_attention == 'y':
                        flagged=1
                    else:
                        flagged=0
                    indices_a = [i for i, x in enumerate(labels) if x == a]
                    sum_counts=0
                    X_train_clusters[str(a)]=pd.DataFrame({})
                    for index_a in indices_a:
                        s=unique_relationships_profile.iloc[index_a]
                        mask = (X_test == s).all(axis=1)
                        for i in range(0,sum([int(x) for x in mask.values.tolist()])):
                            rowForTrainingData=pc[1][index_a].tolist()
                            rowForTrainingData.extend([math.dist(pc[1][index_a], [0,0]), pc[0], flagged])
                            trainingData2.loc[len(trainingData2)] = rowForTrainingData
                    trainingData2["ItemType"] = trainingData2["ItemType"].astype("category").cat.codes
                        
                        sum([int(x) for x in mask.values.tolist()])
                        X_train_clusters[str(a)]=pd.concat([X_train_clusters[str(a)], 
                            X_train[mask]], ignore_index=True)    
                          
                    """
                        flagged_clusters=""
                        while flagged_clusters not in ["a", "b"]:
                            flagged_clusters=input("Which of these clusters contain broken items?\n")
                        for flagged_cluster in flagged_clusters:
                            flagged_item_types=list(biggest_differences.index[biggest_differences==True])
                            flagged_item_types=input(f"What are the missing/unneccessary item types that are causing the problem in this cluster? {list(biggest_differences.index[biggest_differences==True])}\n") or str(list(biggest_differences.index[biggest_differences==True]))
                            matching_rows[flagged_cluster]['Flagged'] = 1
                            matching_rows[flagged_cluster]['ItemTypesForInspection'] = flagged_item_types
                            trainingData2['Flagged']=1
                    """
                    #trainingData=pd.concat([trainingData, matching_rows['a']], ignore_index=False)
                    #trainingData=pd.concat([trainingData, matching_rows['b']], ignore_index=False)
                    #trainingData = trainingData.drop_duplicates()
                    #trainingData = trainingData.replace({np.nan: 0})

 else:
    print("THERE ARE ONLY TWO CLUSTERS")  
    biggest_differences=(abs(all_unique_relationships[pc[0]].iloc[0]
                    -all_unique_relationships[pc[0]].iloc[1])==5.0)
    if len(biggest_differences.index[biggest_differences==True])>0:
                    #print(f"{a}, {b}")
                    print((f"DIFFERENCE: {all_unique_relationships[pc[0]].iloc[0].name}, "
                        f"{all_unique_relationships[pc[0]].iloc[1].name}"))
                    indices_a = [i for i, x in enumerate(labels) if x == 0]
                    indices_b = [i for i, x in enumerate(labels) if x == 1]
                    matching_rows_a = all_relationships[pc[0]][all_relationships[pc[0]].eq(all_unique_relationships[pc[0]].iloc[0]).all(axis=1)]
                    print(f"CLUSTER {a}, {len(matching_rows_a)} ITEMS")
                    print(matching_rows_a)
                    matching_rows_b = all_relationships[pc[0]][all_relationships[pc[0]].eq(all_unique_relationships[pc[0]].iloc[1]).all(axis=1)]
                    print(f"CLUSTER {b}, {len(matching_rows_b)} ITEMS")
                    print(matching_rows_b)
                    print(biggest_differences.index[biggest_differences==True])         



trainingData = trainingData[~trainingData.index.duplicated(keep=False)] 
save_versioned_pickle_file(trainingData2, 'training_data2', folder='../projects/am2_project/data')


X=trainingData2.drop_duplicates().drop(['Flagged', 'ItemType'], axis=1)
dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
X=pd.DataFrame(trainingData3['Distance'])
y=trainingData3[['Flagged']]
dtc.fit(X, y)
X["ItemType"] = X["ItemType"].astype("category").cat.codes
print(df)


X=pd.DataFrame(trainingData2.drop_duplicates()['Distance'])
y=trainingData2.drop_duplicates()['Flagged']

X_2=pd.DataFrame(trainingData2.drop_duplicates()['Distance'])
y_2=trainingData2.drop_duplicates()['Flagged']

scores = cross_val_score(dtc, X, y, cv=5)

WHAT IF YOU MODIFIED THE TEST DATA SO MOST OF THEM WERE BROKEN? WOULD THE DTC STILL WORK?

a=trainingData[trainingData['ItemType']=='Series'].copy()
y=a[['Flagged']]
X=a.drop(['Flagged', 'ItemType'], axis=1)
X["ItemType"] = X["ItemType"].astype("category").cat.codes
print(a)
dtc.fit(X_2, y_2)

plt.figure(figsize=(10, 10))
tree.plot_tree(
    dtc,
    class_names=["2", "3"],
    filled=True,
    feature_names=['ItemType', 'Distance']
)
plt.show(block=False)


                    all_relationships[pc[0]].iloc[indices_b]


                    

indices = [i for i, x in enumerate(labels) if x == 1]
all_relationships[pc[0]].iloc[indices]



                    d=biggest_differences.index[biggest_difference
                    s==True]

all_relationships[pc[0]]

(abs(all_relationships[pc[0]].iloc[labels.tolist().index(0)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)
biggest_differences=(abs(all_relationships[pc[0]].iloc[labels.tolist().index(0)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)

a=abs(all_relationships[pc[0]].iloc[labels.tolist().index(a)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(b)])
                
for a, b in combinations(set(labels.tolist()), 2):  
                biggest_differences=(abs(all_relationships[pc[0]].iloc[labels.tolist().index(0)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)
                if len(biggest_differences.index[biggest_differences==True])>0:
                    print(f"{a}, {b}")
                    print((f"DIFFERENCE: {all_relationships[pc[0]].iloc[labels.tolist().index(a)].name}, "
                        f"{all_relationships[pc[0]].iloc[labels.tolist().index(b)].name}"))
                    print(biggest_differences.index[biggest_differences==True][0])
                    distances.append(math.dist(x, [0,0]))
d=(abs(all_relationships[pc[0]].iloc[labels.tolist().index(labels.tolist().index(0))]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)

                        all_relationships[item_type].iloc[0]
            
biggest_differences=abs(all_relationships['Instrument'].iloc[19]-all_relationships['Instrument'].iloc[0])==5.0
for x in range(0, optimum_K):
    print(unique_relationships_profile.iloc[labels.tolist().index(x)])
    for item_type in all_relationships.keys():
        for a in range(0, max(labels)):
            for b in range(0, max(labels)):
                all_relationships[item_type].iloc[0]
        

all_principal_components['Data Collection']
distances=[]
for x in all_principal_components['Data Collection']:
    distances.append(math.dist(x, [0,0]))
top5_indices = np.argsort(distances)[-5:]
all_relationships[pc[0]]

# 0 and 9 are in different clusters...    
biggest_differences=unique_relationships_profile.iloc[0]-unique_relationships_profile.iloc[9]==5.0
biggest_differences.index[biggest_differences==True]
unique_relationships_profile

for a, b in combinations(set(labels.tolist()), 2):
                print(f"{a}, {b}")    
                biggest_differences=abs(all_relationships['Instrument'].iloc[labels.tolist().index(a)]-all_relationships['Instrument'].iloc[labels.tolist().index(b)])==5.0
                biggest_differences.index[biggest_differences==True]


for a, b in combinations(set(labels.tolist()), 2):    
        biggest_differences=unique_relationships_profile.iloc[a]-unique_relationships_profile.iloc[b]==5.0
        biggest_differences.index[biggest_differences==True]