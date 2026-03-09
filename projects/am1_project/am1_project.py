from sklearn.model_selection import train_test_split
import pandas as pd
from ml_resources import (
    read_dataset_from_file, 
    filter_values_by_length,
    convert_dictionary_to_dataframe,
    apply_pipeline)

PROCESS: 
1. GET DATA FROM COLECTICA

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py

#item_topics=read_dataset_from_file('../data/item_topics_1.pickle')
#question_da=read_dataset_from_file('../data/all_question_categories_1.pickle')
#all_question_summaries=read_dataset_from_file('../data/usoc_question_summaries_1.pickle')

am1_data=read_dataset_from_file('../projects/am1_project/data/am1_data_1.pickle')

2. PERFORM QUALITY CONTROL, E.G. REMOVE DATA WITH MISSING, INADEQUATE values, ARRAYS with
PARTICULAR VALUES

# We now run quality control code to filter items which don't meet certain criteria. 
# E.g. the question summary is too short, the question has fewer than N categories
# associated with it, the question summary contains text, a question has a set of
# categories associated with it that are not deemed to have predictive value, e.g. yes/no

filtered_questions=filter_values_by_length(am1_data, "Summary", 10)

filtered_questions_by_number_of_categories=filter_values_by_length(am1_data, 
    "QuestionCategories", 
    3, 
    filter_type="less_than")

3. Convert the JSON dictionary to an dataframe that is suitable for use with pipelines etc

df=convert_dictionary_to_dataframe(filtered_questions)
df = df.dropna(subset=["Topic"])
df.index = range(0, len(df))     
#transformed_embeddings = apply_pipeline(df, 'Summary')
class_proportions = df["Topic"].value_counts()
print(class_proportions)
print(class_proportions[class_proportions<10])
# We need to remove questions that have topics for which there are less than two instances,
# in order for the stratified splitting performed by train_test_split
# to be possible
questions_with_unique_topics=list(class_proportions[class_proportions<2].index)
df=df[~df['Topic'].isin(questions_with_unique_topics)]
# Need to recalculate indexes, so the pipeline operations will work...
df.index = range(0, len(df))


level_one_topics=pd.DataFrame([x[0:3] for x in df["Topic"]])
level_one_class_proportions = level_one_topics.value_counts()
print(level_one_class_proportions)


4. PERFORM DATA TRANSFORMATIONS EXCLUDING CALCULATION OF EMBEDDINGS EG CONVERT ARRAYS
TO ORDERED LOWER CASE STRING
4a. TRANSFORM TEXT COLUMNS TO EMBEDDINGS

transformed_embeddings = apply_pipeline(df, 'Summary')
#pd.DataFrame({"embeddings": list(pipeline.fit_transform(df))})


5. split data into training and test

y=transformed_embeddings['topic']
X=transformed_embeddings.drop('topic', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test set
    random_state=42,     # for reproducibility
    stratify=y           # ensures balanced class proportions
)

pipeline_input=pd.DataFrame([x[1]['Summary'] for x in am1_data['uk.iser.ukhls'].items()], columns=['question_summaries'])

5. CREATE DATASET WITH PARTICULAR SET OF COLUMNS EG DROP THE RAW TEXT THAT WE HAVE created
EMBEDDINGS FROM


