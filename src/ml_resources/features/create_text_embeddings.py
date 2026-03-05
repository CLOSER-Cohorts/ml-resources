from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embedding_from_item(agencyId, identifier, item_text, item_embeddings={}):
    if agencyId not in item_embeddings.keys():
            item_embeddings[agencyId] = {}
    item_embeddings[agencyId][identifier] = model.encode(item_text)