from pydantic import BaseModel
from typing import List, Optional, Dict

class Layer(BaseModel):
    block_id: str
    params: Optional[Dict] = {}

class NetworkDefinition(BaseModel):
    name: str
    description: Optional[str] = ""
    layers_sequence: List[Layer]