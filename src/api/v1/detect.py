from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from typing import List

import cv2
import numpy as np
from io import BytesIO
from models.detect import DetectRequest

from utils import modelManager, configManager
from PIL import Image
import json

detectRouter = APIRouter()

@detectRouter.post("/detect")
async def detect(
    config: str = Form(...), 
    images: List[UploadFile] = File(...)
):
    try:
        configData = json.loads(config)
        configObj = DetectRequest(**configData)

        # 存在しないモデル名が指定された時
        if not configObj.model:
            configObj.model = configManager.config.get('detection', 'defaultModel')
        elif not (configObj.model in configManager.config['models']):
            raise HTTPException(status_code=400, detail="model not found in config")

        imageDatas = []
        for image in images:
            imageBinary = await image.read()
            imageArray = np.frombuffer(imageBinary, np.uint8)
            imageDatas.append(cv2.imdecode(imageArray, cv2.IMREAD_COLOR))

        # imageData = Image.open(BytesIO(imageBinary))
            
        model = modelManager.getModel(configObj.model)
        results = model(
            imageDatas,
            conf=configObj.confidenceThreshold or configManager.config.get('detection', 'defaultConfidenceThreshold'),
            iou=configObj.nmsThreshold or configManager.config.get('detection', 'defaultNmsThreshold'),
            imgsz=640,
            half=configManager.config.getboolean('detection', 'half'),
            device=configManager.config.get('detection', 'device')
        )

        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     result.show()  # display to screen
        #     result.save(filename="result.jpg")  # save to disk

        # Assuming results[0] contains the image in numpy format
        imgArray = results[0].plot()  # Get the numpy array of the image
        # print(results)
        for result in results:
            print(result.boxes)
        # imgPil = Image.fromarray(imgArray)  # Convert numpy array to PIL Image

        _, imgEncoded = cv2.imencode('.jpg', imgArray)
        # エンコードした画像をメモリ内に保存
        imgIO = BytesIO(imgEncoded.tobytes())

        return StreamingResponse(imgIO, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
