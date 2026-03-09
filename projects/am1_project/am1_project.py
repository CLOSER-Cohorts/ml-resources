from ml_resources import read_dataset_from_file, filter_values_by_length

PROCESS: 
1. GET DATA FROM COLECTICA

# We assume that the code in get_data_from_colectica.py has already been executed, and
# that there are pickle files in the data directory containing question summaries,
# categories, topics, etc. This code is in get_data_from_colectica.py

#item_topics=read_dataset_from_file('../data/item_topics_1.pickle')
#question_da=read_dataset_from_file('../data/all_question_categories_1.pickle')
#all_question_summaries=read_dataset_from_file('../data/usoc_question_summaries_1.pickle')

am1_data=read_dataset_from_file('../data/am1_data_1.pickle')


2. PERFORM QUALITY CONTROL, E.G. REMOVE DATA WITH MISSING, INADEQUATE values, ARRAYS with
PARTICULAR VALUES

# We now run quality control code to filter items which don't meet certain criteria. 
# E.g. the question summary is too short, the question has fewer than N categories
# associated with it, the question summary contains text, a question has a set of
# categories associated with it that are not deemed to have predictive value, e.g. yes/no

filtered_questions_by_number_of_categories=filter_values_by_length(am1_data, 
    "QuestionCategories", 
    3, 
    filter_type="less_than")
filtered_questions_by_length_of-summary=filter_values_by_length(am1_data, 
    "Summary", 
    10, 
    filter_type="less_than")

3. PERFORM DATA TRANSFORMATIONS EXCLUDING CALCULATION OF EMBEDDINGS EG CONVERT ARRAYS
TO ORDERED LOWER CASE STRING
4. TRANSFORM TEXT COLUMNS TO EMBEDDINGS
5. CREATE DATASET WITH PARTICULAR SET OF COLUMNS EG DROP THE RAW TEXT THAT WE HAVE created
EMBEDDINGS FROM


