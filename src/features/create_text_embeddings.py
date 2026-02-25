from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def createEmbeddingFromItem(agencyId, identifier, item_text, item_embeddings={}):
    if agencyId not in item_embeddings.keys():
            item_embeddings[agencyId] = {}
    item_embeddings[agencyId][identifier] = model.encode(item_text)

all_question_embeddings={}
for agencyId in all_question_summaries.keys():
    print(f"Creating question embeddings for {agencyId}...")
    for questionSummary in all_question_summaries[agencyId].items():
        createEmbeddingFromItem(agencyId, questionSummary[0], questionSummary[1], all_question_embeddings)

for agencyId in all_question_summaries.keys():
    print(f"{agencyId}, {len(list(all_question_embeddings[agencyId].items()))}, {len(list(all_question_summaries[agencyId].items()))}")