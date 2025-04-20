# Dockerfile for serving the FastAPI app

# Use official Python slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y 

# Copy requirements and install Python dependencies
COPY requirements.txt ./
COPY setup.py ./
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 

# Copy source code, API, and MLflow runs (if needed for local serving)
COPY src/ ./src/
COPY api/ ./api/
COPY mlruns/ ./mlruns/

RUN pip install -e .

# Expose FastAPI port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]