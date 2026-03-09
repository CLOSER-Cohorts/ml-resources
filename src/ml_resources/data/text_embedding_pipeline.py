from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from ml_resources import read_dataset_from_file
import pandas as pd

question_summaries=read_dataset_from_file('../data/usoc_summaries_1.pickle')
embedding_transformer = FunctionTransformer(lambda x: model.encode(x.squeeze()), validate=True)
preprocessor = ColumnTransformer([
    ("Embedding", embedding_transformer, ["question_summaries"])
],
)
pipeline = Pipeline([("feature_creation", preprocessor)])
data = pd.DataFrame(question_summaries['uk.iser.ukhls'].values(), columns=['question_summaries')
transformed_embeddings = pd.DataFrame({"embeddings": list(pipeline.fit_transform(data))})
