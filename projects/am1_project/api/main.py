# from repo root: python -m uvicorn projects.am1_project.api.main:app --reload
#
# sample request body:
"""
{
    "items":[
    {"TextLabel": "How often do you play sports?", "ItemCategories": "every day every week twice a week", "ItemType": "Question", "AgencyId": "uk.wchads","HasCategories": "yes"},
    {"TextLabel": "How often do you play sports?", "ItemCategories": "every day every week twice a week", "ItemType": "Variable", "AgencyId": "uk.alspac","HasCategories": "yes"}
    ]
}
"""

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Literal
import pickle
import numpy as np
import pandas as pd
import time
import logging
import json
import mlflow
import mlflow.sklearn
from src.ml_resources import (
    apply_pipeline)
from src.logging.utility import StructuredMessage, setup_logging
from projects.am1_project.src.utility import convert_df_to_ndarray

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")



#YOU NEED TO CHANGE THE CODE HERE SO IT IS GETTING THE LIVE VERSION OF THE MODEL FROM
#MLFLOW
with open("./config/config.json") as f:
            general_config = json.load(f)
mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
mlflow_client = mlflow.MlflowClient()
trainedModel = mlflow.sklearn.load_model(
        model_uri="models:/logistic_regression_for_topic_classification@live")
                    
#modelfile = open('./projects/am1_project/model/trainedModelAllStudies/trainedModelAllStudies_1.pickle', 'rb')
#trainedModel = pickle.load(modelfile)
categories=trainedModel.classes_.tolist()

class Item(BaseModel):
    TextLabel: str | None = None
    ItemCategories: str | None = None
    ItemType: Literal["question", "variable"] | None = None
    AgencyId: Literal['uk.iser.ukhls',
        'uk.cls.bcs70', 
        'uk.cls.mcs', 
        'uk.wchads', 
        'uk.lha',
        'uk.alspac', 
        'uk.mrcleu-uos.sws', 
        'uk.cls.nextsteps', 
        'uk.genscot', 
        'uk.mrcleu-uos.heaf', 
        'uk.whitehall2', 
        'uk.mrcleu-uos.hcs', 
        'uk.cls.ncds'] | None = None
    HasCategories: Literal["yes", "no"] | None = None
    @field_validator("HasCategories", mode="before")
    @classmethod
    def normalize_yes_no(this_class, field_value):
        if isinstance(field_value, str):
            return field_value.strip().lower()
        return field_value
    @field_validator("ItemType", mode="before")
    @classmethod
    def normalize_item_type(this_class, field_value):
        if isinstance(field_value, str):
            return field_value.strip().lower()
        return field_value

class InferenceRequest(BaseModel):
    items: list[Item]
    
class PredictionResponse(BaseModel):
    predictions: list[str]

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": trainedModel is not None
    }

@app.post("/categorise_items/")
async def categorise_items(api_request: InferenceRequest):
    start_time = time.time()
    df = pd.DataFrame([item.model_dump() for item in api_request.items])
    df["ItemType"] = df["ItemType"].map({"question": 1, "variable": 0})
    df["HasCategories"] = df["HasCategories"].map({"yes": 1, "no": 0})
    df["AgencyId"] = df["AgencyId"].map({'uk.iser.ukhls' : 0, 
        'uk.cls.bcs70' : 1, 
        'uk.cls.mcs' : 2, 
        'uk.wchads' : 3, 
        'uk.lha' : 4, 
        'uk.alspac' : 5, 
        'uk.mrcleu-uos.sws' : 6, 
        'uk.cls.nextsteps' : 7, 
        'uk.genscot' : 8, 
        'uk.mrcleu-uos.heaf' : 9, 
        'uk.whitehall2' : 10, 
        'uk.mrcleu-uos.hcs' : 11, 
        'uk.cls.ncds' : 12})
    print(df)
    transformed_embeddings = apply_pipeline(df, ['TextLabel', 'ItemCategories'], training=False)
    X = convert_df_to_ndarray(transformed_embeddings)
    """
    X = np.hstack([
     np.vstack(transformed_embeddings['summary_embeddings']),
     np.vstack(transformed_embeddings['category_embeddings']),
     np.vstack(transformed_embeddings['item_type']),
     np.vstack(transformed_embeddings['has_categories'])
    ])
    """
    #result = trainedModel.predict(X)
    results=trainedModel.predict_proba(X)
    predictions=[]
    confidence_scores=[]
    for result in results:
        #result.tolist().index(max(result))
        confidence_scores.append(max(result).item())
        predictions.append(categories[result.tolist().index(max(result))])
    latency = time.time() - start_time
    logger.info(StructuredMessage(message='Categorise text',
        application="am1",
        operation_type="item_classification",
        number_of_predictions=len(transformed_embeddings),
        labels=df['TextLabel'].values.tolist(),
        size_of_labels=df['TextLabel'].memory_usage(deep=True),
        categories=df['ItemCategories'].values.tolist(),
        size_of_categories=df['ItemCategories'].memory_usage(deep=True),
        item_types=df['ItemType'].values.tolist(),
        latency=latency,
        results=predictions,
        confidence=confidence_scores
        ))
    print(predictions)
    return PredictionResponse(predictions=predictions)
    