from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def createEmbeddingFromItem(agencyId, identifier, item_text, item_embeddings={}):
    if agencyId not in item_embeddings.keys():
            items_embeddings[agencyId] = {}
    items_embeddings[agencyId][identifier] = model.encode(questionText)
