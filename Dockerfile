FROM python:3.9-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ ./models/

# Copy YOLO model file
COPY yolo11n-pose.pt .

# Copy application code
COPY app/ ./app/
COPY app.py .
COPY utils.py .
COPY train.py .

# Set environment variables
ENV PORT=8080

# Run with Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app 