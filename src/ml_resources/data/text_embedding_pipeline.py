from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from ml_resources import read_dataset_from_file
import pandas as pd

question_summaries=read_dataset_from_file('../data/usoc_summaries_1.pickle')
embedding_transformer = FunctionTransformer(lambda x: model.encode(x.squeeze()), validate=False)
preprocessor = ColumnTransformer([
    ("Embedding", embedding_transformer, ["question_summaries"])
],
)
pipeline = Pipeline([("feature_creation", preprocessor)])
transformed_embeddings = pd.DataFrame({"embeddings": list(pipeline.fit_transform(data))})

#d = pd.DataFrame(am1_data['uk.iser.ukhls'].values(), columns=['question_summaries')
pipeline_input=pd.DataFrame([x[1]['Summary'] for x in am1_data['uk.iser.ukhls'].items()], columns=['question_summaries'])
