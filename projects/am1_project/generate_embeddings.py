from ml_resources import ( create_embedding_from_item, read_dataset_from_file)
from ml_resources import save_versioned_pickle_file

# We assume that the question summaries have been created and saved as in 
# get_data_from_colectica.property, and are available in a pickle file...

all_question_summaries=read_dataset_from_file('../data/usoc_question_summaries_1.pickle')

all_question_embeddings={}
for index, agencyId in enumerate(all_question_summaries.keys()):
    for index, question_summary in enumerate(all_question_summaries[agencyId].items()):
        print((f"Creating question embeddings for {agencyId}. " 
            f"Processing item {index} of {len(all_question_summaries[agencyId].keys())}"))
        create_embedding_from_item(agencyId, 
            question_summary[0], 
            question_summary[1], 
            all_question_embeddings)

# Assuming that we got question summaries for the uk.lha and uk.iser.ukhls studies and 
# calculated their embeddings, here is how we would save them into a versioned pickle file... 
save_versioned_pickle_file(all_question_embeddings, 'all_question_embeddings', folder='../data')

