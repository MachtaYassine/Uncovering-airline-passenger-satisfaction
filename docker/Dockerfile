# Dockerfile for serving the FastAPI app

# Use official Python slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
COPY setup.py ./
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy source code, API, and MLflow runs (if needed for local serving)
COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data


RUN pip install -e .

RUN train --model-type torch_nn --hidden-dim 64 --epochs 20 --lr 0.001
RUN train --model-type random_forest --n-estimators 50
RUN train --model-type logistic_regression --max-iter 100


# Expose FastAPI port
EXPOSE 8000

# Default command to run the FastAPI app
CMD ["python", "api/run_api.py", "--reload"]