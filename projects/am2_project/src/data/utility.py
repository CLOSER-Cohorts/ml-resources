from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml_resources import (
    obtain_correctly_labelled_data,
    create_model_package,
    save_versioned_pickle_file )
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import math

def create_am2_input_features(items, colectica_client):    
    df_relationships = pd.DataFrame()
    df_relationships_unique = pd.DataFrame()
    # Using 'enumerate(items)' to create an index may be slow due to the complexity
    # of the item objects
    count = 0
    for item in items:
      print(f"{colectica_client.item_code_inv(item['ItemType'])}")
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
                else:
                    newRow[x] = 1.0
        # Ensure all columns in new_row exist in df
        for key in newRow:
            if key not in df_relationships.columns:
                df_relationships[key] = 0 
        #df_relationships.loc[len(df_relationships)] = newRow
        df_relationships.loc[f"{item['AgencyId']}:{item['Identifier']}"] = newRow
        df_relationships = df_relationships.replace({np.nan: 0})
    return df_relationships

def generate_data_for_classification(item_type, 
    pca_data,
    all_data,
    clf,
    dataset_name="data",
    graphs_directory='./projects/am2_project/graphs/'):
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
    plt.savefig(f"{graphs_directory}isolation_forest_outliers_{dataset_name}_{item_type}.png")
    plt.close()
    for index, target_variable in enumerate(target_variables):
            row_for_final_dataset = pca_data[index].tolist()
            row_for_final_dataset.extend([math.dist(pca_data[index], [0,0]), item_type, target_variable])
            final_dataset.loc[len(final_dataset)] = row_for_final_dataset
    return final_dataset

def generate_pca_data(X_train,
    item_type,
    dataset_name="_data",
    all_principal_components={},
    graphs_directory='./projects/am2_project/graphs/'):
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
        plt.savefig(f"{graphs_directory}pca_{dataset_name}_{item_type}.png")
        plt.close()
        return principalComponents

def train_semi_supervised_model(model,
    df_relationships_unique,
    item_types,
    all_data,
    all_principal_components={},
    dataset_name="_",
    generate_classification_report=False):
    for item_type in item_types:
        if input(f"Do you want to process the {item_type} items? ") in ['y', 'Y']:
            test_size_value=input("What proportion of the items do you want to put aside for testing? ")
            # Generate training data, and train a decision tree
            X_train, X_test = train_test_split(
                df_relationships_unique, # this needs to be specific to data type it's not at present
                test_size=float(test_size_value)
            )
            pca_data=generate_pca_data(X_train,
                item_type, 
                dataset_name=f"{dataset_name}_training", 
                all_principal_components=all_principal_components)
            clf = IsolationForest(max_samples=100, random_state=0)
            clf.fit(pca_data)
            training_dataset_isolation_forest=generate_data_for_classification(item_type,
                pca_data,
                X_train,
                clf,
                dataset_name=dataset_name)
            model_name_version=f"{item_type}_classifier_for_error_detection"
            training_data_description=f"{len(X_train)}_{item_type}_items"
            human_labelled_training_data=obtain_correctly_labelled_data(training_dataset_isolation_forest,
                'We are testing isolation forests',
                'Flagged',
                model_name_version,
                training_data_description,
                item_type=item_type,
                target_variable_is_binary=True,
                categories=[-1,1]
                )
            all_data[item_type]=human_labelled_training_data
            X=pd.DataFrame(human_labelled_training_data[['x', 'y', 'ItemType', 'Distance']])
            X["ItemType"] = X["ItemType"].astype("category").cat.codes
            y=human_labelled_training_data[['Flagged']]
            print("FITTING MODEL")
            print(X)
            print(y)
            model.fit(X, y)
            # Generate test data, and calculate the accuracy of the decision tree
            print("Now we will run tests on the data we set aside...")
            input_for_test_pca=X_test.drop_duplicates().fillna(0)
            test_pca_data = generate_pca_data(input_for_test_pca,
                item_type,
                dataset_name=f"{dataset_name}_test",
                all_principal_components=all_principal_components)
            test_dataset = generate_data_for_classification(item_type,
                test_pca_data,
                X_test,
                clf,
                dataset_name="test_data")
            test_dataset.index = input_for_test_pca.index
            test_dataset["ItemType"] = test_dataset["ItemType"].astype("category").cat.codes
            y_pred=model.predict(pd.DataFrame(test_dataset[['x', 'y', 'ItemType', 'Distance']]))
            # We replace the 'Flagged' target variable in the test dataset which contains values
            # calculated by an IsolationForest with the predictions produced by the supervised
            # model.
            test_dataset['Flagged']=y_pred
            print(test_dataset)
            obtain_correctly_labelled_data(test_dataset,
                'We are testing isolation forests',
                'Flagged',
                model_name_version,
                training_data_description,
                item_type=item_type,
                target_variable_is_binary=True,
                only_relabel_outliers=False,
                categories=[-1,1],
                generate_classification_report=generate_classification_report
                )
            notes=input("Write any notes you want to include in the model metadata here, or press 'Enter' to leave the notes field empty. ")    
            model_package=create_model_package(model,
                X,
                'Flagged', 
                preprocessing=["PCA"],
                notes=notes,
                model_version=model_name_version,
                training_data_version=training_data_description)
            save_versioned_pickle_file(model_package,
                model_name_version, 
                folder='./projects/am2_project/models')