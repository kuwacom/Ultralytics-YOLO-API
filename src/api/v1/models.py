from typing import List
from fastapi import APIRouter, HTTPException
import utils
import json

modelsRouter = APIRouter()

@modelsRouter.get("/models", response_model=List[str])
async def detect():
    try:
        return list(utils.modelManager.Models.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
