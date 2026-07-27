from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_model
from src.dataset import get_transforms, IMAGENET_MEAN, IMAGENET_STD

app = FastAPI(title="Weather Classification API")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = {0: "rain", 1: "fog", 2: "snow"}
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

models_list = []


@app.on_event("startup")
def load_models():
    """Загружаем все модели ансамбля один раз при старте сервиса."""
    transform = get_transforms(IMAGENET_MEAN, IMAGENET_STD)['val']
    app.state.transform = transform

    for fold in range(5):
        path = os.path.join(MODELS_DIR, f'model_fold_{fold}.pth')
        model = create_model(num_classes=3, freeze_backbone=True, device=DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        model.eval()
        models_list.append(model)

    print(f"Загружено моделей: {len(models_list)}")


@app.get("/")
def root():
    return {"status": "ok", "models_loaded": len(models_list)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = app.state.transform(image).unsqueeze(0).to(DEVICE)

    all_probs = None
    with torch.no_grad():
        for model in models_list:
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            all_probs = probs if all_probs is None else all_probs + probs

    all_probs /= len(models_list)
    pred_idx = int(torch.argmax(all_probs, dim=1).item())
    confidence = float(all_probs[0, pred_idx].item())

    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(all_probs[0, i].item()), 4)
            for i in range(3)
        }
    }