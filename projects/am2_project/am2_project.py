from colectica_api import ColecticaObject
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.ml_resources.data import colectica_utility
from src.ml_resources import read_dataset_from_file, save_versioned_pickle_file
colectica_client = colectica_utility.C
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import combinations
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest

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

def find_clusters(pca_data):
    potential_k_values=range(2,min(10,len(pca_data)))
    scores = []
    for k in range(2,min(10,len(pca_data))):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(pca_data)
            score = silhouette_score(pca_data, labels)
            scores.append(score)
    optimum_K = potential_k_values[scores.index(max(scores))]
    kmeans = KMeans(optimum_K)
    labels = kmeans.fit_predict(pca_data)
    return labels

def generate_data_for_classification(item_type, pca_data, all_data, clf, data_name="data"):
    target_variables=clf.predict(pca_data)
    unique_data_rows = all_data.drop_duplicates()
    final_dataset=pd.DataFrame({}, columns=['x', 'y', 'Distance', 'ItemType', 'Flagged'])
    # CREATE THE PLOT WITH THE OUTLIERS
    scatter2 = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=target_variables, s=20, edgecolor="k")
    labels=["outliers", "inliers"]
    handles, labels = scatter2.legend_elements()
    plt.axis("square")
    plt.legend(handles=handles, labels=labels, title="true class")
    plt.title("Outlier detection for questionnaire instruments")
    plt.show(block=False)
    plt.savefig(f"isolation_forest_outliers_{data_name}_{item_type}_2.png")
    plt.close()
    for index, target_variable in enumerate(target_variables):
            row_for_final_dataset = pca_data[index].tolist()
            row_for_final_dataset.extend([math.dist(pca_data[index], [0,0]), item_type, int(target_variable)])
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
    pca_data = PCA(n_components=2)
    unique_relationships_profile = X_train.drop_duplicates().fillna(0)
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
        plt.savefig(f"pca_{item_type}_{dataset_name}_2.png")
        plt.close()
        return principalComponents

item_types = [
'Data File',
'Instrument',
'Data Collection'
]

allData={}
for item_type in item_types:
    if input(f"Do you want to process the {item_type} items? ") in ['y', 'Y']:
        test_size_value=input("What proportion of the items do you want to put aside for testing? ")
        # Generate training data, and train a decision tree
        input_for_pca=df_relationships.drop_duplicates().fillna(0)
        X_train, X_test = train_test_split(
            input_for_pca, # this needs to be specific to data type it's not at present
            test_size=float(test_size_value)
        )
        pca_data=generate_pca_data(X_train, item_type, "training")
        clf = IsolationForest(max_samples=100, random_state=0)
        clf.fit(pca_data)
        final_dataset=generate_data_for_classification(item_type, pca_data, X_train, clf)
        dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        model_name_version=f"{item_type}_decision_tree_classifier_for_error_detection_v1"
        training_data=f"{len(X_train)}_{item_type}_items"
        X=pd.DataFrame(final_dataset[['x', 'y', 'ItemType', 'Distance']])
        X["ItemType"] = X["ItemType"].astype("category").cat.codes
        allData[item_type]=final_dataset
        y=final_dataset[['Flagged']]
        dtc.fit(X, y)
        # Generate test data, and calculate the accuracy of the decision tree
        print("Now we will run tests on the data we set aside...")
        input_for_test_pca=X_test.drop_duplicates().fillna(0)
        test_pca_data = generate_pca_data(input_for_test_pca, item_type, "test")
        test_dataset = generate_data_for_classification(item_type, test_pca_data, X_test, clf, data_name="test_data")
        test_dataset.index = input_for_test_pca.index
        test_dataset["ItemType"] = test_dataset["ItemType"].astype("category").cat.codes
        y_pred=dtc.predict(pd.DataFrame(test_dataset[['x', 'y', 'ItemType', 'Distance']]))
        





        # THE BELOW CODE CAN BE USEFUL FOR ADDING TO THE CODE THAT IS USED TO CHECK PERFORMANCE
        indices_flagged = [i for i, x in enumerate(y_pred) if x == 1]
        y_test=test_dataset['Flagged']
        y_test.values.tolist().extend(y_pred.tolist())
        target_values=[str(x) for x in set(y_test) | set(y_pred)]
        report = classification_report(y_test, y_pred, target_names=target_values)
        print(report)
        notes_on_experiment = input("Enter any notes on this experiment you wish to record (e.g. parameters, evaluation metrics, what did/didn't work): ")
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

save_versioned_pickle_file(trainingData2, 'training_data2', folder='../projects/am2_project/data')


WHAT IF YOU MODIFIED THE TEST DATA SO MOST OF THEM WERE BROKEN? WOULD THE DTC STILL WORK?


plt.figure(figsize=(10, 10))
tree.plot_tree(
    dtc,
    class_names=["2", "3"],
    filled=True,
    feature_names=['ItemType', 'Distance']
)
plt.show(block=False)

(abs(all_relationships[pc[0]].iloc[labels.tolist().index(0)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)
biggest_differences=(abs(all_relationships[pc[0]].iloc[labels.tolist().index(0)]
                    -all_relationships[pc[0]].iloc[labels.tolist().index(1)])==5.0)

