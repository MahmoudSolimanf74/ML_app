# File: backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import pandas as pd
import io

from schemas import ManualPredictionRequest, PredictionResponse, CSVPredictionResponse
from model import predict_single, predict_batch
from utils import validate_csv_columns

REQUIRED_COLUMNS = [
    "Company","TypeName","Ram","Weight","Price_euros","touch_screen","IPS_display","PPI","CPU","SSD","HDD","GPU_brand","OS"
]

app = FastAPI(title="Laptop Price Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/predict/manual", response_model=PredictionResponse, summary="Manual prediction")
async def predict_manual(payload: ManualPredictionRequest):
    try:
        price = predict_single(payload.dict())
        return {"price": price}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/csv", response_model=CSVPredictionResponse, summary="CSV batch predictions")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file or unreadable content")

    missing = validate_csv_columns(df, REQUIRED_COLUMNS)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required columns: {', '.join(missing)}"
        )

    try:
        predictions = predict_batch(df)
        return {"prediction": predictions}
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate predictions")