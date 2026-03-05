from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from ml_resources import read_dataset_from_file
import pandas as pd

def create_embedding_from_item(agencyId, identifier, item_text, item_embeddings={}):
    if agencyId not in item_embeddings.keys():
            item_embeddings[agencyId] = {}
    item_embeddings[agencyId][identifier] = model.encode(item_text)

# Below is code that creates a pipeline object for creating embeddings 

question_summaries=read_dataset_from_file('../data/usoc_summaries_1.pickle')
embedding_transformer = FunctionTransformer(lambda x: model.encode(x.squeeze()), validate=True)
preprocessor = ColumnTransformer([
    ("Embedding", embedding_transformer, ["question_summaries"])
],
)
pipeline = Pipeline([("feature_creation", preprocessor)])
data = pd.DataFrame(question_summaries['uk.iser.ukhls'].values(), columns=['question_summaries'])
transformed_embeddings = pd.DataFrame({"embeddings": list(pipeline.fit_transform(data))})
