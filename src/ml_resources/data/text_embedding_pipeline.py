from sentence_transformers import SentenceTransformer, util
#model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from src.ml_resources import read_dataset_from_file
import pandas as pd

embedding_length=384

"""
all_dummy_embeddings==encode_columns(lr_model_data['X_train'], ['TextLabel', 'ItemCategories'])
all_dummy_embeddings2=encode_columns_narrow(lr_model_data['X_train'][0:150], ['TextLabel', 'ItemCategories'])
lr_wide2.fit(embeddings.drop(['AgencyId', 'TextLabel', 'ItemCategories', 'ItemType', 'HasCategories'], axis=1), lr_model_data['y_train'])
probs = lr_wide2.predict_proba(embeddings_test.drop(['AgencyId', 'TextLabel', 'ItemCategories', 'ItemType', 'HasCategories'], axis=1))

def encode_columns(df, columns):
    out = df.copy()
    for col in columns:
        out[col + "_emb"] = model.encode(df[col].tolist())
    return out
"""

def encode_columns(df, columns):
    out = df.copy()
    for col in columns:
        embeddings = model.encode(df[col].tolist())
        emb_df = pd.DataFrame(
            embeddings,
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])],
            index=df.index
        )
        out = pd.concat([out, emb_df], axis=1)
    return out

def encode_columns_narrow(df, columns):
    out = df.copy()
    for col in columns:
        embeddings = model.encode(df[col])
        out['embedding'] = list(embeddings)
    return out


def apply_pipeline(data, columns, training=True):
    embedding_transformer = FunctionTransformer(lambda x: model.encode(x.iloc[:, 0].tolist()), validate=False)
    column_transformers=[]
    for column in columns:
        column_transformers.append(
        ("Embedding_"+column, embedding_transformer, [column]))
    preprocessor = ColumnTransformer(
        column_transformers,
    remainder="passthrough"
    )
    pipeline = Pipeline([("feature_creation", preprocessor)])
    transformed_data=list(pipeline.fit_transform(data))
    if training:
        topic_slice = [x[embedding_length*2+3] for x in transformed_data]
    else:
        topic_slice = None
    transformed_embeddings = pd.DataFrame({"summary_embeddings": [x[0:embedding_length] for x in transformed_data],
     "category_embeddings": [x[embedding_length:embedding_length*2] for x in transformed_data],
     "item_type": [x[embedding_length*2] for x in transformed_data],
     "agency_id": [x[embedding_length*2+1] for x in transformed_data],
     "has_categories": [x[embedding_length*2+2] for x in transformed_data]
     },
     index=data.index
    )
    if training:
        transformed_embeddings["topic"] = topic_slice
    return transformed_embeddings