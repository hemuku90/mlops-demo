FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY src/app src/app
COPY src/model src/model
# Model is not baked into the image in production patterns
# It will be served via Seldon Core pulling from MLflow/S3

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/wine_model

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
