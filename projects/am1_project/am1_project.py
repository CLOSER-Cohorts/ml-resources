from sklearn.model_selection import train_test_split
import pandas as pd
from ml_resources import (
    read_dataset_from_file,
    save_versioned_pickle_file, 
    filter_values_by_length,
    convert_dictionary_to_dataframe,
    apply_pipeline,
    train_model,
    create_model_data_object,
    calculate_accuracy)

#PROCESS: 
#1. GET DATA FROM COLECTICA

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py

am1_data=read_dataset_from_file('../projects/am1_project/data/am1_data_1.pickle')

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

transformed_embeddings = read_dataset_from_file('../projects/am1_project/data/transformed_embeddings_1.pickle')

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
input_feature_list=[]
for input_feature in ['summary_embeddings', 'category_embeddings']:
        input_feature_list.append(np.vstack(data_for_model['X_test'][input_feature]))
X_test = np.hstack(
        input_feature_list
)
trainedModel.predict(X_test)
predictions_with_probabilities=trainedModel.predict_proba(X_test)
wrong_predictions=calculate_accuracy(trainedModel,
    predictions_with_probabilities,
    X_test,
    lr_model_data['y_test'].values,
    N=3)
