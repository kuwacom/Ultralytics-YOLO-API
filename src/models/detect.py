from pydantic import BaseModel
from typing import Optional, Dict, List

class DetectRequest(BaseModel):
    model: str
    confidenceThreshold: Optional[float] = 0.5
    nmsThreshold: Optional[float] = 0.4
    detectedImage: bool = False

class DetectResponse(BaseModel):
    model: str
    confidenceThreshold: Optional[float]
    nmsThreshold: Optional[float]
    namesCount: Dict[int, int]
    detectedImage: str = ""
    detectTime: Dict
    xywh: List[List[float]]
    xywhn: List[List[float]]
    xyxy: List[List[float]]
    xyxyn: List[List[float]]


