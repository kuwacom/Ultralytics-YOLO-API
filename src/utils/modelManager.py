from ultralytics import YOLO

Models = {}

def getModel(modelName):
    if modelName in Models:
        return Models[modelName]
    else:
        return None

def addModel(modelName, model):
    Models[modelName] = model

def initModels(models):
    for modelName, modelPath in models:
        addModel(modelName, YOLO(modelPath))