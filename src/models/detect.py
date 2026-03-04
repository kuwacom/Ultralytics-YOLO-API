from pydantic import BaseModel
from typing import Optional, Dict, List

class DetectRequest(BaseModel):
    model: str
    confidenceThreshold: Optional[float] = 0.5
    nmsThreshold: Optional[float] = 0.4
    detectedImage: bool = False

class DetectionResult(BaseModel):
    classId: int
    className: str
    confidence: float
    xywh: List[float]
    xywhn: List[float]
    xyxy: List[float]
    xyxyn: List[float]

class DetectResponse(BaseModel):
    model: str
    confidenceThreshold: Optional[float]
    nmsThreshold: Optional[float]
    namesCount: Dict[int, int]
    detections: List[DetectionResult]
    detectedImage: str = ""
    detectTime: Dict
    # 旧フォーマットの互換性
    # xywh: List[List[float]]
    # xywhn: List[List[float]]
    # xyxy: List[List[float]]
    # xyxyn: List[List[float]]


