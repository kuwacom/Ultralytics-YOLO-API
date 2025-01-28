from pydantic import BaseModel
from typing import Dict

class Model(BaseModel):
    name: str
    names: Dict[int, str]
