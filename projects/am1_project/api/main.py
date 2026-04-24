# from repo root: python -m uvicorn projects.am1_project.api.main:app --reload

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import time
import logging
from src.ml_resources import (
    apply_pipeline)
from src.logging.utility import StructuredMessage, setup_logging

logger=setup_logging(project="am1_project", log_file="logs/am1_log.json")

#from src.ml_resources import read_dataset_from_fil
# v
# e

modelfile = open('./projects/am1_project/model/trainedModelAllStudies/trainedModelAllStudies_1.pickle', 'rb')
trainedModel = pickle.load(modelfile)
categories=trainedModel.classes_.tolist()
"""
testDataFile = open('./projects/am1_project/data/testData/testData_1.pickle', 'rb')
testData = pickle.load(testDataFile)

X = np.hstack([
     np.vstack(testData['summary_embeddings']),
     np.vstack(testData['category_embeddings'])
 ])
"""

class Item(BaseModel):
    TextLabel: list[str] | None = None
    ItemCategories: list[str] | None = None
    ItemType: list[int] | None = None
    HasCategories: list[int] | None = None
    
app = FastAPI()

#message_value = str(trainedModel.predict(X))

@app.get("/")
async def root():
    return {"message": "message_value"}


@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

"""
{"Summary": ["olly", "ben"], "QuestionCategories": ["100", "dog"]}
"""
@app.post("/categorise_questions/")
async def categorise_questions(item: Item):
    start_time = time.time()
    df = pd.DataFrame(item.dict())
    transformed_embeddings = apply_pipeline(df, ['TextLabel', 'ItemCategories'])
    X = np.hstack([
     np.vstack(transformed_embeddings['summary_embeddings']),
     np.vstack(transformed_embeddings['category_embeddings']),
     np.vstack(transformed_embeddings['item_type']),
     np.vstack(transformed_embeddings['has_categories'])
    ])
    #result = trainedModel.predict(X)
    results=trainedModel.predict_proba(X)
    predictions=[]
    confidence_scores=[]
    for result in results:
        #result.tolist().index(max(result))
        confidence_scores.append(max(result).item())
        predictions.append(categories[result.tolist().index(max(result))])
    latency = time.time() - start_time
    print(result)
    print(transformed_embeddings['item_type'])
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
    return list(predictions)
    

    
    
    #question_summary=item['Summary']
    #question_categories=item['Categories']
    #transformed_embeddings = apply_pipeline(df, ['Summary', 'QuestionCategories'])