from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import List

import cv2
import torch
import json
import numpy as np
from models.detect import DetectRequest, DetectResponse
from utils import modelManager, configManager, converter

detectRouter = APIRouter()

@detectRouter.post("/detect", response_model=List[DetectResponse])
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
            imgsz=640, # 処理する際の画像解像度
            half=configManager.config.getboolean('detection', 'half'),
            device=configManager.config.get('detection', 'device')
        )

        # 結果は公式documentを参照
        # https://docs.ultralytics.com/modes/predict/#working-with-results
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     result.show()  # display to screen
        #     result.save(filename="result.jpg")  # save to disk

        detectResponse: List[DetectResponse] = []
        for result in results:
            # 検出されたタグクラス名の量をオブジェクトで表現 例: {0: 2, 12: 2} 
            names, counts = torch.unique(result.boxes.cls, return_counts=True)
            namesCount = dict(zip(
                names.to(torch.int).tolist(),
                counts.tolist()
            ))

            print(namesCount)

            base64Image = ""
            if configObj.detectedImage:
                base64Image = converter.ndarray2base64(result.plot())
            
            detectResponse.append(DetectResponse(
                model=configObj.model,
                confidenceThreshold=configObj.confidenceThreshold,
                nmsThreshold=configObj.nmsThreshold,
                namesCount=namesCount,
                detectedImage=base64Image,
                detectTime=result.speed,
                xywh=result.boxes.xywh.tolist(),
                xywhn=result.boxes.xywhn.tolist(),
                xyxy=result.boxes.xyxy.tolist(),
                xyxyn=result.boxes.xyxyn.tolist(),
            ))

        return detectResponse
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
