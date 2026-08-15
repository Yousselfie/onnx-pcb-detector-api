import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "models/best.onnx"
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not per request
    ml["model"] = YOLO(MODEL_PATH, task="detect")
    yield
    ml.clear()

app = FastAPI(title="PCB Component Detector", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    results = ml["model"](img, verbose=False)
    r = results[0]

    detections = [
        {
            "class_name": r.names[int(b.cls[0])],
            "confidence": round(float(b.conf[0]), 4),
            "bbox_xyxy": [round(float(x), 1) for x in b.xyxy[0].tolist()],
        }
        for b in r.boxes
    ]
    return {"count": len(detections), "detections": detections}
