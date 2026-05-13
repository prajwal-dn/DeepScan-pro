FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for OpenCV and some python packages)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Run the API
CMD ["python", "deepfake_detector.py", "api"]
