from typing import List
from fastapi import APIRouter, HTTPException
from utils import modelManager
from models.models import Model

modelsRouter = APIRouter()

@modelsRouter.get("/models", response_model=List[Model])
async def detect():
    try:
        responseList = []
        for modelName in modelManager.Models.keys():
            responseList.append(Model(
                name=modelName,
                names=modelManager.getModel(modelName).names
            ))
        return responseList
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
