from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from src.ml_resources import (
    apply_pipeline)

#from src.ml_resources import read_dataset_from_fil
# 
# e
"""
modelfile = open('./projects/am1_project/model/trainedModel/trainedModel_1.pickle', 'rb')
trainedModel = pickle.load(modelfile)
testDataFile = open('./projects/am1_project/data/testData/testData_1.pickle', 'rb')
testData = pickle.load(testDataFile)

X = np.hstack([
     np.vstack(testData['summary_embeddings']),
     np.vstack(testData['category_embeddings'])
 ])
"""

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()

#message_value = str(trainedModel.predict(X))

@app.get("/")
async def root():
    return {"message": message_value}


@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict

@app.post("/categorise_questions/")
async def categorise_questions(item: Item):
    return item
    

    
    
    #question_summary=item['Summary']
    #question_categories=item['Categories']
    #transformed_embeddings = apply_pipeline(df, ['Summary', 'QuestionCategories'])