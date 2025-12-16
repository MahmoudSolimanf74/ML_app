# File: backend/app/schemas.py
from pydantic import BaseModel

class ManualPredictionRequest(BaseModel):
    Company: str
    TypeName: str
    Ram: int
    Weight: float
    touch_screen: bool
    IPS_display: bool
    PPI: float
    CPU: str
    SSD: int
    HDD: int
    GPU_brand: str
    OS: str

class PredictionResponse(BaseModel):
    price: int

class CSVPredictionResponse(BaseModel):
    prediction: list[int]
