FROM python:3.12-slim

# Install system deps for torch
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/service_requirements.txt
RUN pip install --no-cache-dir -r /app/service_requirements.txt

# Copy app code
WORKDIR /app
COPY . /app

# Expose Render’s $PORT
CMD uvicorn model_service:app --host 0.0.0.0 --port $PORT
