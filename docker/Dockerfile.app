# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir --default-timeout=100 -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Create a non-root user with home directory
RUN groupadd -r appuser && useradd -r -m -g appuser appuser

# Copy installed packages from builder
# We copy to the user's home directory.
COPY --from=builder /root/.local /home/appuser/.local

# Update PATH and PYTHONPATH
ENV HOME=/home/appuser
ENV PATH=$HOME/.local/bin:$PATH
# Explicitly add the user site-packages to PYTHONPATH
ENV PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:/app
ENV MODEL_PATH=/app/models/wine_model
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY src/app src/app
COPY src/model src/model

# Set ownership for the application code and ensure home is owned by appuser
RUN chown -R appuser:appuser /app && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
