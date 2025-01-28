from ultralytics import YOLO
from typing import Dict, Optional

Models: Dict[str, YOLO] = {}

def getModel(modelName: str):
    if modelName in Models:
        return Models[modelName]
    else:
        return None

def addModel(modelName: str, model: YOLO):
    Models[modelName] = model

def initModels(models: list[tuple[str, str]]):
    for modelName, modelPath in models:
        addModel(modelName, YOLO(modelPath))