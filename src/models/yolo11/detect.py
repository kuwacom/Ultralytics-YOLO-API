from pydantic import BaseModel
from typing import Optional

class DetectRequest(BaseModel):
    model: str
    confidenceThreshold: Optional[float] = 0.5
    nmsThreshold: Optional[float] = 0.4
