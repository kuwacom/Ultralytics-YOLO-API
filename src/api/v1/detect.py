from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from models.yolo11.detect import DetectRequest

from utils import modelManager, configManager
from PIL import Image

detectRouter = APIRouter()

@detectRouter.post("/detect")
async def detect(
    config: DetectRequest, 
    image: UploadFile = File(...)
):
    try:
        # 存在しないモデル名が指定された時
        if not config.model:
            config.model = configManager.config.get('detection', 'defaultModel')
        elif not (config.model in configManager.config):
            raise HTTPException(status_code=400, detail="model not found in config")
        
        imageData = await image.read()
        
        model = modelManager.getModel(config.model)
        results = model.predict(
            imageData,
            conf=config.confidenceThreshold or configManager.config.get('detection', 'defaultConfidenceThreshold'),
            iou=config.nmsThreshold or configManager.config.get('detection', 'defaultNmsThreshold'),
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
        imgArray = results[0].numpy  # Get the numpy array of the image
        imgPil = Image.fromarray(imgArray)  # Convert numpy array to PIL Image

        imgIO = BytesIO()
        imgPil.save(imgIO, format='JPEG')  # Save the PIL Image to BytesIO
        imgIO.seek(0)

        # 結果の画像をバイナリとして返す
        # imgIO = BytesIO()
        # results[0].save(imgIO, 'JPEG')
        # imgIO.seek(0)

        return StreamingResponse(imgIO, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
