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
- A `best.onnx` model file placed at `models/best.onnx` — export it from the [training repo](https://github.com/Yousselfie/pcb-component-detection-yolo26) with:


