from sklearn.model_selection import train_test_split
from ml_resources import ( update_dataset,
    add_input_feature_to_dataset,
    read_dataset_from_file, 
    create_model_data_object, 
    read_dataset_from_file,
    train_model,
    calculate_accuracy,
    create_embedding_from_item,
    save_versioned_pickle_file,
    create_embedding_from_item
)
from ml_resources.data import colectica_utility
import pandas as pd

colectica_client = colectica_utility.C
all_studies = colectica_client.search_items(colectica_client.item_code('Series'), SearchLatestVersion=True)['Results']
all_question_summaries={}
all_variable_labels={}
# Get question summaries for one study...
colectica_utility.get_questions_for_studies(all_studies[0], all_question_summaries)

# Get question summaries for all studies...
colectica_utility.get_questions_for_studies(all_studies, all_question_summaries)

all_question_embeddings={}
for agencyId in all_question_summaries.keys():
    print(f"Creating question embeddings for {agencyId}...")
    for question_summary in all_question_summaries[agencyId].items():
        create_embedding_from_item(agencyId, question_summary[0], question_summary[1], all_question_embeddings)

# Show that we have an equal number of question summaries and question embeddings for all
# studies, i.e. we have embeddings for all summaries...
for agencyId in all_question_summaries.keys():
    print(f"{agencyId}, {len(list(all_question_embeddings[agencyId].items()))}, {len(list(all_question_summaries[agencyId].items()))}")

item_topics={}
for study in [all_studies[0]]:
    study_search_set = [{
                 "agencyId": study['AgencyId'],
                 "identifier": study['Identifier'],
                 "version": study['Version']
                }]
    print(f"Getting topics for items in {study['AgencyId']}...")
    study_question_items = colectica_client.search_items(colectica_client.item_code('Question'),
        SearchSets=study_search_set,
        SearchLatestVersion=True)['Results']
    get_topics_for_items(study_question_items, colectica_client, topics=item_topics)

# Assuming that we got question summaries for the uk.lha and uk.iser.ukhls studies and 
# calculated their embeddings, and obtained the topics assigned to those questions, here 
# is how we would save them into a versioned pickle file... 
save_versioned_pickle_file(all_question_embeddings, 'lha_usoc_question_embeddings', folder='../data')
save_versioned_pickle_file(item_topics, 'lha_usoc_topics', folder='../data')

# Assuming that we have already saved question embeddings and their associated topics into 
# pickle files, here is how we read them from the pickle files...
item_topics=read_dataset_from_file('../data/lha_usoc_topics_1.pickle')
all_question_embeddings=read_dataset_from_file('../data/lha_usoc_question_embeddings_1.pickle')

# The code below shows how we get the model data for uk.lha, uk.user.ukhls and those two
# studies combined, create and train a prediction model, and measure that model's accuracy...

identifiers_usoc = list(item_topics['uk.iser.ukhls'].keys())
identifiers_lha = list(item_topics['uk.lha'].keys())

# Create a dataset...

dataset = update_dataset('uk.lha',
    identifiers_lha,
    all_question_embeddings['uk.lha'], 
    'QuestionEmbedding', 
    'Topic', 
    item_topics['uk.lha'])

# ...and now we update the dataset we just created so it includes data from another study...

dataset = update_dataset('uk.iser.ukhls',
    identifiers_usoc, 
    all_question_embeddings['uk.iser.ukhls'], 
    'QuestionEmbedding', 
    'Topic',
    item_topics['uk.iser.ukhls'])

X_train, X_test, y_train, y_test = train_test_split(
    pd.concat([dataset['uk.lha']['InputFeatures'], dataset['uk.iser.ukhls']['InputFeatures']]),
    pd.concat([dataset['uk.lha']['Targets'], dataset['uk.iser.ukhls']['Targets']]),
    train_size=0.9)
X_train, X_test, y_train, y_test = train_test_split(dataset['uk.lha']['InputFeatures'],
    dataset_lha['Targets'], train_size=0.9 )    
X_train, X_test, y_train, y_test = train_test_split(dataset['uk.iser.ukhls']['InputFeatures'],
    dataset_usoc['Targets'], train_size=0.9 )    

lr_model_data=create_model_data_object(X_train, X_test, y_train, y_test)
trainedModel=train_model(lr_model_data, selected_input_features='QuestionEmbedding')
trainedModel.predict(list(lr_model_data['X_test']['QuestionEmbedding'].values))
predictions_with_probabilities=trainedModel.predict_proba(list(lr_model_data['X_test']['QuestionEmbedding'].values))
wrong_predictions=calculate_accuracy(trainedModel,
    predictions_with_probabilities, 
    lr_model_data['X_test']['QuestionEmbedding'].values, 
    lr_model_data['y_test'].values,
    N=3)

all_question_categories=colectica_utility.get_categories_for_questions('uk.iser.ukhls', identifiers_usoc)
save_versioned_pickle_file(all_question_categories, 'all_question_categories', folder='../data')

# If we have already saved the question categories to a pickle file, we can read them from that...

all_question_categories=read_dataset_from_file('../data/all_question_categories_1.pickle')

all_question_category_embeddings={}
for agencyId in all_question_categories.keys():
    print(f"Creating question category embeddings for {agencyId}...")
    for question_categories in all_question_categories[agencyId].items():
        create_embedding_from_item(agencyId,
            question_categories[0],
            question_categories[1],
            all_question_category_embeddings)

save_versioned_pickle_file(all_question_category_embeddings, 'all_question_category_embeddings', folder='../data')

# If we have already saved the question categories embeddings to a pickle file, we can
# read them from that...

all_question_category_embeddings=read_dataset_from_file('../data/all_question_category_embeddings_1.pickle')

add_input_feature_to_dataset('uk.iser.ukhls',
    identifiers_usoc,
    all_question_category_embeddings,
    "QuestionCategoryEmbeddings",
    dataset)

save_versioned_pickle_file(dataset, 'questions_dataset', folder='../data')