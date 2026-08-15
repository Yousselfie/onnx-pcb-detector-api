FROM python:3.11-slim

# System libs needed by OpenCV (pulled by ultralytics)  at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgl1 \
	libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# installing deps first before copying app code
# docker caches this so editing code later doesn't run pip again
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copying app and model
COPY app/ app/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
