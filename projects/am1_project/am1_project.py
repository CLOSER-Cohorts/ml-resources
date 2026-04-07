from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import json
from src.ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file, 
    filter_values_by_length,
    convert_dictionary_to_dataframe,
    apply_pipeline,
    train_model,
    create_model_data_object,
    calculate_accuracy)
from src.ml_resources.data import colectica_utility

#PROCESS: 
#1. GET DATA FROM COLECTICA

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py.
# We assume the code is being run from the repository root directory (ml-resources).

am1_data=read_dataset_from_file('./projects/am1_project/data/am1_data_4.pickle')

# We can check if there is newly available data...
with open("./projects/am1_project/config/am1_config.json") as f:
    project_config = json.load(f)

new_am1_data={}
colectica_utility.get_questions_in_containing_items(project_config['Studies'], new_am1_data, "Summary")

question_keys_from_repository_for_dataset = []
for agencyId in new_am1_data.keys():
    question_keys_from_repository_for_dataset.extend(new_am1_data[agencyId].keys())

question_keys_from_current_dataset = []
for agencyId in am1_data.keys():
    question_keys_from_current_dataset.extend(am1_data[agencyId].keys())

new_question_identifiers=check_for_newly_available_data_am1(question_keys_from_repository_for_dataset,
    question_keys_from_current_dataset)

# have not yet implemented what to do with new questions
for agency in new_am1_data.keys():
    print(agency)
    print([x for x in new_question_identifiers if x in new_am1_data[agency].keys()])

#2. PERFORM QUALITY CONTROL, E.G. REMOVE DATA WITH MISSING, INADEQUATE values, ARRAYS with
# PARTICULAR VALUES

# We now run quality control code to filter items which don't meet certain criteria. 
# E.g. the question summary is too short, the question has fewer than N categories
# associated with it, the question summary contains text, a question has a set of
# categories associated with it that are not deemed to have predictive value, e.g. yes/no

filtered_questions=filter_values_by_length(am1_data, "Summary", 10)

filtered_questions_by_number_of_categories=filter_values_by_length(filtered_questions,
    "QuestionCategories", 
    3)

filtered_questions=filtered_questions_by_number_of_categories

#3. Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc

df=convert_dictionary_to_dataframe(filtered_questions)

#4. Now we perform some data cleaning. Resetting the index is important so the pipeline
# operations will work (the indices have to be continuous numeric values with no gaps)

df = df.dropna(subset=["Topic"]).reset_index(drop=True)
     
# View the proportion of topics in our data
class_proportions = df["Topic"].value_counts()
print(class_proportions)
print(class_proportions[class_proportions<10])

# We need to remove questions that have topics for which there are less than two instances,
# in order for the stratified splitting performed by train_test_split
# to be possible.

questions_with_unique_topics=list(class_proportions[class_proportions<2].index)
df=df[~df['Topic'].isin(questions_with_unique_topics)].reset_index(drop=True)

# Investigate the proportions of topics at level one. For now we won't do anything
# based on this, but we may use this information at a further point.
level_one_topics=pd.DataFrame([x[0:3] for x in df["Topic"]])
level_one_class_proportions = level_one_topics.value_counts()
print(level_one_class_proportions)

#5. PERFORM DATA TRANSFORMATIONS. TRANSFORM TEXT COLUMNS TO EMBEDDINGS

# If we have already calculated the embeddings and saved them to a pickle file, we read them in...

transformed_embeddings = read_dataset_from_file('./projects/am1_project/data/transformed_embeddings_1.pickle')

# ...otherwise we can calculate them from scratch, and save them to a file...

transformed_embeddings = apply_pipeline(df, ['Summary', 'QuestionCategories'])

save_versioned_pickle_file(transformed_embeddings, 'transformed_embeddings', folder='../projects/am1_project/data')

#6. split data into training and test

y=transformed_embeddings['topic']
X=transformed_embeddings.drop('topic', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test set
    random_state=42,     # for reproducibility
    stratify=y           # ensures balanced class proportions
)

#7 Create a logistic regression model, test it, and measure it's accuracy

lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
trainedModel=train_model(lr_model_data, selected_input_features=['summary_embeddings', 'category_embeddings'])

# Save the model...

save_versioned_pickle_file(trainedModel, 'trainedModel', folder='../projects/am1_project/model')
save_versioned_pickle_file(testData, 'testData', folder='../projects/am1_project/data')


input_feature_list=[]
for input_feature in ['summary_embeddings', 'category_embeddings']:
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

#b=obtain_correct_data_labels(final_dataset_copy, "do this", "Flagged")

report = classification_report(lr_model_data['y_test'].values, y_pred, target_names=set(lr_model_data['y_test'].values))
print(report)
model_name_version=f"logistic_regression_for_topic_classification_v1"
training_data=f"Summaries and categories for {len(X_train)} questions"
notes_on_experiment = input("Enter any notes on this experiment you wish to record (e.g. parameters, evaluation metrics, what did/didn't work): ")
with open(f"classification_report_topic_classification_v1.txt", "w") as f:
    _ = f.write(f"Classification report for topic classification logistic regression model\n\n")
    _ = f.write(f"Notes: \n{notes_on_experiment}\n\n")
    _ = f.write(f"Model name and version: \n{model_name_version}\n\n")
    _ = f.write(f"Training data: \n{training_data}\n\n")
    _ = f.write(f"Classification report: \n{report}\n\n")
    _ = f.write("\nThese items have been flagged as needing attention.\n\n")
    # wrong_predictions may not contain the desired information at present,
    # we may need to update the calculate_accuracy function
    for wrong_prediction in wrong_predictions:
        _ = f.write(wrong_prediction)

cm = confusion_matrix(y_test, y_pred)
print(cm)
target_names=list(set(lr_model_data['y_test'].values))
for index, topic in enumerate(target_names):
    print(f"TOPIC {topic}, {sum(cm[index,])}")
    for index2, x in enumerate(cm[index,]):
        if x>0:
            print(f"{target_names[index2]}: {x}")

for index, x in enumerate(cm):
    print(cm[index,])