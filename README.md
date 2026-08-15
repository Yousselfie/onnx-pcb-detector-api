# ONNX PCB Detector API

## A production-style deployment pipeline that serves the fine-tuned YOLO26 PCB component detector as a REST API.

## Problem
Training a model is only half the job — to be useful it has to be served. This project wraps the PCB component detector in a deployable inference service: a containerized HTTP API that any client can POST an image to and receive structured detections, with automated build-and-test on every change.

## Approach
There are four stages to this deployment pipeline:

1. **Export to ONNX**: the trained best.pt is exported to an open, frameword-independent format, ONNX, with a runtime that gives up to ~3x faster CPU inference than native PyTorch. This speed improvement is important for a CPU-hosted API.
2. **Serve with FastAPI**: a lightweight option to load the model once at startup and expose a /predict endpoint that accepts an uploaded images and returns detections as JSON.
3. **Containerize with Docker**: the service is packaged into a Docker image so it runs the same on any machine.
4. **CI with GitHub Actions**: every pushed change rebuilds the image and runs a test again /health.

## API
 
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check → `{"status": "ok"}` |
| `/predict` | POST | Upload an image (form field `file`) → detections JSON |
| `/docs` | GET | Interactive Swagger UI |
 
Example `/predict` response:
 
```json
{
  "count": 2,
  "detections": [
    {"class_name": "resistor", "confidence": 0.91, "bbox_xyxy": [12.4, 88.1, 44.0, 121.7]},
    {"class_name": "capacitor", "confidence": 0.86, "bbox_xyxy": [150.2, 60.5, 190.8, 102.3]}
  ]
}
```
 
## Reproduce
 
### Prerequisites
- Python 3.10+
- Docker (for the containerized version)

1. Obtain the `best.onnx` model file to be placed at `models/best.onnx` — export it from the [training repo](https://github.com/Yousselfie/pcb-component-detection-yolo26) by opening the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bZJOhZKFngtrtlRCCIYqq38GcLp_oQaW?usp=sharing)
and running it on a CPU runtime (**Runtime** -> **Change runtime type** -> *CPU**)
*Ensure you change the GDrive paths in the code to where your files live in your own drive before you run*
2. Move the exported `best.onnx` file into `app/`
3. Running the model:
### (no Docker)
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000/docs and try `/predict`, or:
```bash
curl -X POST http://localhost:8000/predict -F "file=@test_pcb.jpg"
```
### (with Docker)
```bash
docker build -t pcb-detector-api .
docker run -p 8000:8000 pcb-detector-api
```

