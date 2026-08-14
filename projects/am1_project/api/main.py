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

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator
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
    read_dataset_from_file,
    apply_pipeline)
from src.logging.utility import StructuredMessage, setup_logging
from projects.am1_project.src.utility import convert_df_to_ndarray
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

limiter = Limiter(key_func=get_remote_address)

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")

MAX_ITEMS_PER_REQUEST = 100
MAX_TEXT_LENGTH = 2000
#YOU NEED TO CHANGE THE CODE HERE SO IT IS GETTING THE LIVE VERSION OF THE MODEL FROM
#MLFLOW
with open("./config/config.json") as f:
            general_config = json.load(f)
mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
mlflow_client = mlflow.MlflowClient()

trained_models={}
trained_models['uk.iser.ukhls'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@ukhls"
        )
trained_models['uk.whitehall2'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@whitehall2"
        )
trained_models['uk.cls.nextsteps'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@nextsteps"
        )
trained_models['uk.lha'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@lha"
        )
trained_models['uk.wchads'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@wchads"
        )
trained_models['uk.cls.bcs70'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@bcs70"
        )
trained_models['uk.alspac'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@alspac"
        )
"""
trained_models['uk.mrcleu-uos.sws'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@sws"
        )
trained_models['uk.mrcleu-uos.heaf'] = mlflow.sklearn.load_model(
        model_uri="models:/Logistic Regression for topic classification@heaf"
        )
"""



#modelfile = open('./projects/am1_project/model/trainedModel/Logistic Regression Model_242.pickle', 'rb')
trainedModel = read_dataset_from_file('./projects/am1_project/model/tunedModelAllStudies/tunedModelAllStudies_1.pickle')
categories=trainedModel.classes_.tolist()

class Item(BaseModel):
    TextLabel: str = Field(..., max_length=MAX_TEXT_LENGTH)
    ItemCategories: str | None = Field("", max_length=MAX_TEXT_LENGTH)
    ItemType: Literal["question", "variable"]
    """
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
        'uk.cls.ncds']
    """
    #HasCategories: Literal["yes", "no"]
    #@field_validator("HasCategories", mode="before")
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
    items: list[Item] = Field(..., min_length=1, max_length=MAX_ITEMS_PER_REQUEST)
    
#class PredictionResponse(BaseModel):
#    predictions: list[str]
#    confidence_scores: list[list[float]]

class PredictionProbability(BaseModel):
    label: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: list[str]
    top_5_predictions: list[list[PredictionProbability]]

app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)
app.add_middleware(SlowAPIMiddleware)

@app.get("/")
async def root():
    return {"message": "API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": trainedModel is not None
    }

@app.post("/categorise_items/{agency_id}")
@limiter.limit("2/minute")
async def categorise_items(request: Request, 
   agency_id:str, 
   api_request: InferenceRequest):
    start_time = time.time()
    df = pd.DataFrame([item.model_dump() for item in api_request.items])
    print(df)
    df["ItemType"] = df["ItemType"].map({"question": 1, "variable": 0})
    #df["HasCategories"] = df["HasCategories"].map({"yes": 1, "no": 0})
    """
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
    """
    print(df)
    #feature_columns=['ItemType', 'TextLabel_embeddings', 'ItemCategories_embeddings']
    feature_columns=['item_type', 'summary_embeddings', 'category_embeddings']
    transformed_embeddings = apply_pipeline(df, ['TextLabel', 'ItemCategories'],
        training=False)
    print(transformed_embeddings.iloc[0])
    X = convert_df_to_ndarray(transformed_embeddings, input_features=feature_columns)
    """
    X = np.hstack([
     np.vstack(transformed_embeddings['summary_embeddings']),
     np.vstack(transformed_embeddings['category_embeddings']),
     np.vstack(transformed_embeddings['item_type']),
     np.vstack(transformed_embeddings['has_categories'])
    ])
    """
    #result = trainedModel.predict(X)
    results=trained_models[agency_id].predict_proba(X)
    predictions=[]
    low_confidence_predictions = []
    confidence_scores={}
    confidence_scores_list=[]
    top_5_predictions = []
    for result in results:
        #result.tolist().index(max(result))
        prediction=categories[result.tolist().index(max(result))]
        if agency_id not in confidence_scores.keys():
          confidence_scores[agency_id]={}
        if prediction not in confidence_scores[agency_id]:
          confidence_scores[agency_id][prediction]=[] 
        confidence_scores[agency_id][prediction].append(max(result).item())
        top_N_results_indices = np.argsort(result)[-5:][::-1]
        print(top_N_results_indices)
        top_N_results = np.array(categories)[top_N_results_indices]
        confidence=result[top_N_results_indices]
        # reverse so results are in descending order...
        predictions.append(prediction)
        #confidence_scores_list.append(confidence)
        item_predictions = [
            PredictionProbability(
                label=str(categories[i]),
                probability=float(result[i])
            )
            for i in top_N_results_indices
        ]
        top_5_predictions.append(item_predictions)
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
    return PredictionResponse(
       #confidence_scores=confidence_scores_list
       predictions=predictions,
       top_5_predictions=top_5_predictions
       )
    