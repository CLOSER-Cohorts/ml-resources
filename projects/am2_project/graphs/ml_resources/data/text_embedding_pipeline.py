from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from ml_resources import read_dataset_from_file
import pandas as pd

def apply_pipeline(data, column):
    embedding_transformer = FunctionTransformer(lambda x: model.encode(x.squeeze()), validate=False)
    preprocessor = ColumnTransformer([
        ("Embedding", embedding_transformer, [column])
    ],
    remainder="passthrough"
    )
    pipeline = Pipeline([("feature_creation", preprocessor)])
    transformed_data=list(pipeline.fit_transform(data))
    transformed_embeddings = pd.DataFrame({"embeddings": [x[0:384] for x in transformed_data],
        "topic": [x[384] for x in transformed_data]})
    return transformed_embeddings
#df2.index = range(0, len(df2))     
#transformed_embeddings = apply_pipeline(df2, 'Summary')    

#transformed_embeddings = apply_pipeline(df, 'Summary')
#d = pd.DataFrame(am1_data['uk.iser.ukhls'].values(), columns=['question_summaries')
#pipeline_input=pd.DataFrame([x[1]['Summary'] for x in am1_data['uk.iser.ukhls'].items()], columns=['question_summaries'])
