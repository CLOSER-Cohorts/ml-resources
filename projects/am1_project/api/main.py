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
import base64
from cryptography.hazmat.primitives import serialization
from src.cryptography import hash_directory
from cryptography.exceptions import InvalidSignature
import os
import sys
import cryptography
from pathlib import Path
import hashlib

limiter = Limiter(key_func=get_remote_address)

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")

MAX_ITEMS_PER_REQUEST = 100
MAX_TEXT_LENGTH = 2000
model_name = 'Logistic Regression for topic classification'
with open("./config/config.json") as f:
            general_config = json.load(f)
mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
mlflow_client = mlflow.MlflowClient()

def retrieve_and_validate_models(trained_models, model_name, agencies, public_key):
    print("=" * 80)
    print("VALIDATION ENVIRONMENT")
    print("Python:", sys.executable)
    print("Python:", sys.version)
    print("MLflow:", mlflow.__version__)
    print("Cryptography:", cryptography.__version__)
    print("CWD:", os.getcwd())
    print(sys.executable)
    print(os.getcwd())
    print(Path("./keys/ed25519_public_key.pem").resolve())
    key_path = Path("./keys/ed25519_public_key.pem").resolve()
    key_bytes = key_path.read_bytes()
    print("Key path:", key_path)
    print("Key file SHA256:", hashlib.sha256(key_bytes).hexdigest())
    print("Key file size:", len(key_bytes))
    print(
    hashlib.sha256(
        public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo
        )
    ).hexdigest())
    print("=" * 80)
    for agency in agencies:
        alias = agency.rsplit(".", 1)[-1]
        print(agency)
        mv = mlflow_client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )
        model_uri = f"models:/{model_name}@{alias}"
        local_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri
            )
        signature_b64 = mv.tags["ed25519_signature"]
        expected_hash = mv.tags["model_sha256"]
        #model_path = mlflow.artifacts.download_artifacts(
        #    artifact_uri=mv.source
        #)
        signature = base64.b64decode(signature_b64) 
        actual_hash = hash_directory(local_model_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Model hash does not match MLflow metadata"
            )
        try:
            public_key.verify(
                signature,
                bytes.fromhex(actual_hash),
            )
        except InvalidSignature:
            raise RuntimeError(
                "Model cryptographic signature is invalid"
            )
        trained_models[agency] = mlflow.sklearn.load_model(
            local_model_path
            )

trained_models={}
models_to_retrieve=['uk.iser.ukhls',
    'uk.whitehall2',
    'uk.cls.nextsteps',
    'uk.lha',
    'uk.wchads',
    'uk.cls.bcs70',
    'uk.alspac',
    #'uk.mrcleu-uos.sws',
    #'uk.mrcleu-uos.heaf'
    ]

with open("./keys/ed25519_public_key.pem", "rb") as f:
    public_key = serialization.load_pem_public_key(f.read())
    
print(public_key)
retrieve_and_validate_models(trained_models,
    model_name,
    models_to_retrieve,
    public_key)

#modelfile = open('./projects/am1_project/model/trainedModel/Logistic Regression Model_242.pickle', 'rb')
trainedModel = read_dataset_from_file('./projects/am1_project/model/tunedModelAllStudies/tunedModelAllStudies_1.pickle')
#categories=trainedModel.classes_.tolist()

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
    df["ItemType"] = df["ItemType"].map({"question": 0, "variable": 1})
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
    print(trained_models[agency_id].predict(X))
    predictions=[]
    low_confidence_predictions = []
    confidence_scores={}
    confidence_scores_list=[]
    top_5_predictions = []
    for result in results:
        #result.tolist().index(max(result))
        categories=trained_models[agency_id].classes_.tolist()
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
    