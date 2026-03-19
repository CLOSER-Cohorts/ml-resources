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

item_types_string=['Action', 'Archive', 'Category', 'Category Group', 'Category Set', 'ClassificationCorrespondenceTable', 'ClassificationFamily', 'ClassificationIndex', 'ClassificationItem', 'ClassificationLevel', 'ClassificationSeries', 'Code List Group', 'Code List Set', 'Code Set', 'Concept', 'Concept Group', 'Concept Set', 'Conceptual Component', 'Conceptual Variable', 'Conceptual Variable Group', 'Conceptual Variable Set', 'Conditional', 'Control Construct Group', 'Control Construct Set', 'Data Collection', 'Data File', 'Data Layout', 'DataCollection Methodology', 'General Instruction', 'Generation Instruction', 'Individual', 'Instruction Group', 'Instrument', 'Instrument Group', 'Instrument Set', 'Interviewer Instruction', 'Interviewer Instruction Set', 'Logical Product', 'Loop', 'Managed Representation Group', 'Managed Representation Set', 'MeasurementItem', 'MeasurementConstruct', 'Metadata Package', 'NCube', 'NCube Group', 'NCube Set', 'Organization', 'Organization Group', 'Organization Set', 'OtherMaterial', 'OtherMaterialGroup', 'OtherMaterialScheme', 'Physical Data Product', 'Physical Structure', 'PhysicalStructure Set', 'Processing Event', 'Processing Event Group', 'Processing Event Set', 'Processing Instruction Group', 'Processing Instruction Scheme', 'Project', 'Quality Standard', 'Quality Statement', 'Quality Statement Group', 'Quality Statement Set', 'Question', 'Question Activity', 'Question Block', 'Question Grid', 'Question Group', 'Question Set', 'RecordLayout', 'RecordLayout Set', 'Repeat', 'Represented Variable', 'Represented Variable Group', 'Represented Variable Set', 'Reusable Missing Value', 'Sequence', 'Series', 'Statement', 'StatisticalClassification', 'Study', 'SubSeries', 'UnitType', 'UnitTypeScheme', 'UnitTypeGroup', 'Universe', 'Universe Group', 'Universe Set', 'Variable', 'Variable Group', 'Variable Set', 'Variable Statistic', 'While']
item_types_string=['Variable Statistic', 'While']

for item_type in [colectica_client.item_code(x) for x in item_types_string]:
 # if item_type==colectica_client.item_code('Series'):
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
    unique_relationships_profile=df_relationships.drop_duplicates()
    all_unique_relationships[colectica_client.item_code_inv(item_type)] = unique_relationships_profile
    all_relationships[colectica_client.item_code_inv(item_type)] = df_relationships
    pca_data = PCA(n_components=2)
    print(unique_relationships_profile.fillna(0))
    if len(unique_relationships_profile.fillna(0))>1:
        principalComponents = pca_data.fit_transform(
            unique_relationships_profile.fillna(0))
        count=0
        plt.title(f"Plot for {colectica_client.item_code_inv(item_type)}")
        for i in principalComponents:
            plt.scatter(i[0], i[1])
            if True:
                numberInCluster=len(df_relationships[(df_relationships==unique_relationships_profile.iloc[count,:]).all(axis=1)])
                print(numberInCluster)
                plt.text(i[0], i[1], (str(count)+": "+str(numberInCluster)))
            count=count+1
        all_principal_components[colectica_client.item_code_inv(item_type)] = principalComponents
        plt.show(block=False)
        plt.savefig(f"outliers_{colectica_client.item_code_inv(item_type)}.png")
        plt.close()


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

trainingData2=pd.DataFrame({})
for pc in all_principal_components.items():
#for pc in [all_principal_components['Data Collection']]:
 #  if pc[0]=='Series':
 if len(pc[1])>2 and pc[0] in item_types_2:
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
    distances=[]
    coordinates=[]
    for x in set(labels.tolist()):
        distances.append(math.dist(all_principal_components[pc[0]][labels.tolist().index(x)], [0,0]))
        coordinates.append(all_principal_components[pc[0]][labels.tolist().index(x)]).tolist()
    print(distances)
    #for a, b in combinations(set(labels.tolist()), 2):  
    for a in 
                biggest_differences=(abs(all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)]
                    -all_unique_relationships[pc[0]].iloc[labels.tolist().index(b)])==5.0)
                if len(biggest_differences.index[biggest_differences==True])>0:
                    print(f"{a}, {b}")
                    print((f"DIFFERENCE: {all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)].name}, "
                        f"{all_unique_relationships[pc[0]].iloc[labels.tolist().index(b)].name}"))
                    indices_a = [i for i, x in enumerate(labels) if x == a]
                    indices_b = [i for i, x in enumerate(labels) if x == b]
                    matching_rows = {}
                    matching_rows['a'] = all_relationships[pc[0]][all_relationships[pc[0]].eq(all_unique_relationships[pc[0]].iloc[labels.tolist().index(a)]).all(axis=1)]
                    matching_rows['a']['ItemType'] = pc[0]
                    matching_rows['a']['Flagged'] = 0
                    matching_rows['a']['ItemTypesForInspection'] = ""
                    print(f"CLUSTER {a}, {len(matching_rows['a'])} ITEMS")
                    print(matching_rows['a'])
                    matching_rows['b'] = all_relationships[pc[0]][all_relationships[pc[0]].eq(all_unique_relationships[pc[0]].iloc[labels.tolist().index(b)]).all(axis=1)]
                    print(f"CLUSTER {b}, {len(matching_rows['b'])} ITEMS")
                    print(matching_rows['b'])
                    matching_rows['b']['ItemType'] = pc[0]
                    matching_rows['b']['Flagged'] = 0
                    matching_rows['b']['ItemTypesForInspection'] = ""
                    clusters_need_attention="a"
                    trainingData2.loc[len(trainingData2)] = coordinates.extend(pc[0])
                    print(biggest_differences.index[biggest_differences==True])
                    while clusters_need_attention not in ["y", "n"]:
                        clusters_need_attention = input("Are any of these clusters containing broken items? y/n\n")
                    if clusters_need_attention == 'y':
                        flagged_clusters=""
                        while flagged_clusters not in ["a", "b"]:
                            flagged_clusters=input("Which of these clusters contain broken items?\n")
                        for flagged_cluster in flagged_clusters:
                            flagged_item_types=list(biggest_differences.index[biggest_differences==True])
                            flagged_item_types=input(f"What are the missing/unneccessary item types that are causing the problem in this cluster? {list(biggest_differences.index[biggest_differences==True])}\n") or str(list(biggest_differences.index[biggest_differences==True]))
                            matching_rows[flagged_cluster]['Flagged'] = 1
                            matching_rows[flagged_cluster]['ItemTypesForInspection'] = flagged_item_types
                            trainingData2['Flagged']=1
                    trainingData=pd.concat([trainingData, matching_rows['a']], ignore_index=False)
                    trainingData=pd.concat([trainingData, matching_rows['b']], ignore_index=False)
                    #trainingData = trainingData.drop_duplicates()
                    trainingData = trainingData.replace({np.nan: 0})
 else:
    print("THERE ARE ONLY TWO CLUSTERS")  
    biggest_differences=(abs(all_unique_relationships[pc[0]].iloc[0]
                    -all_unique_relationships[pc[0]].iloc[1])==5.0)
    if len(biggest_differences.index[biggest_differences==True])>0:
                    print(f"{a}, {b}")
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
save_versioned_pickle_file(trainingData, 'training_data', folder='../projects/am2_project/data')
dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
y=trainingData[['Flagged']]
X=trainingData.drop(['Flagged', 'ItemTypesForInspection'], axis=1)
X=trainingData.drop(['Flagged'], axis=1)
X["ItemType"] = X["ItemType"].astype("category").cat.codes
print(df)
dtc.fit(X, y)

a=trainingData[trainingData['ItemType']=='Series'].copy()
y=a[['Flagged']]
X=a.drop(['Flagged', 'ItemTypesForInspection'], axis=1)
X["ItemType"] = X["ItemType"].astype("category").cat.codes
print(a)
dtc.fit(X, y)

plt.figure(figsize=(10, 10))
tree.plot_tree(
    dtc,
    class_names=["0", "1"],
    filled=True,
    feature_names=X.columns
)
plt.show()


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