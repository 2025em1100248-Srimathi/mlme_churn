# Dockerfile - runs the FastAPI inference service.
# Build:  docker build -t churn-api .
# Run:    docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY features/ ./features/
COPY serving/ ./serving/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/latest.joblib

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
