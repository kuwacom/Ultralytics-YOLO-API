from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

import cv2
import numpy as np
from io import BytesIO
from models.yolo11.detect import DetectRequest

from utils import modelManager, configManager
from PIL import Image
import json

detectImageRouter = APIRouter()

@detectImageRouter.post("/detectImage")
async def detectImage(
    config: str = Form(...), 
    image: UploadFile = File(...)
):
    try:
        configData = json.loads(config)
        configObj = DetectRequest(**configData)

        # 存在しないモデル名が指定された時
        if not configObj.model:
            configObj.model = configManager.config.get('detection', 'defaultModel')
        elif not (configObj.model in configManager.config['models']):
            raise HTTPException(status_code=400, detail="model not found in config")

        imageBinary = await image.read()
        imageArray = np.frombuffer(imageBinary, np.uint8)
        imageData = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

        # PLIでやるならこれ
        # imageData = Image.open(BytesIO(imageBinary))
            
        model = modelManager.getModel(configObj.model)
        results = model(
            imageData,
            conf=configObj.confidenceThreshold or configManager.config.get('detection', 'defaultConfidenceThreshold'),
            iou=configObj.nmsThreshold or configManager.config.get('detection', 'defaultNmsThreshold'),
            imgsz=640,
            half=configManager.config.getboolean('detection', 'half'),
            device=configManager.config.get('detection', 'device')
        )

        imgArray = results[0].plot()

        _, imgEncoded = cv2.imencode('.jpg', imgArray)
        # エンコードした画像をメモリ内に保存
        imgIO = BytesIO(imgEncoded.tobytes())

        return StreamingResponse(imgIO, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
