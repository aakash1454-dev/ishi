# api/routes/anemia.py

from fastapi import APIRouter, UploadFile, File
from api.services.anemia_model import predict_anemia

router = APIRouter()

@router.post("/predict/anemia")
async def predict_anemia_route(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_anemia(image_bytes)
