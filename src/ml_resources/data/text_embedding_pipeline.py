from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from src.ml_resources import read_dataset_from_file
import pandas as pd

embedding_length=384

def apply_pipeline(data, columns):
    #embedding_transformer = FunctionTransformer(lambda x: model.encode(x.squeeze()), validate=False)
    embedding_transformer = FunctionTransformer(lambda x: model.encode(x.iloc[:, 0].tolist()), validate=False)
    column_transformers=[]
    for column in columns:
        column_transformers.append(
        ("Embedding_"+column, embedding_transformer, [column]))
    print(column_transformers)
    preprocessor = ColumnTransformer(
        column_transformers,
    remainder="passthrough"
    )
    pipeline = Pipeline([("feature_creation", preprocessor)])
    transformed_data=list(pipeline.fit_transform(data))
    transformed_embeddings = pd.DataFrame({"summary_embeddings": [x[0:embedding_length] for x in transformed_data],
     "category_embeddings": [x[embedding_length:embedding_length*2] for x in transformed_data],
     "item_type": [x[embedding_length*2] for x in transformed_data],
     "topic": [x[embedding_length*2+1] for x in transformed_data],
     "has_categories": [x[embedding_length*2+2] for x in transformed_data]}
     )
    return transformed_embeddings