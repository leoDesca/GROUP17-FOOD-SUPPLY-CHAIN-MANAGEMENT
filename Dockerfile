# Dockerfile — Group 17 Supply Chain API (HDBSCAN)
#
# Build:  docker build -t makerere-api .
# Run:    docker run -p 8000:8000 makerere-api
# Test:   curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app

# Install C build tools — the standalone hdbscan package
# compiles Cython extensions and needs gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache the pip layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and code
COPY makerere_Cafeteria_synthetic.csv .
COPY train.py .
COPY test2_model_compression.py .
COPY preprocess.py .
COPY app.py .
COPY templates ./templates
COPY static ./static

# Train the baseline model and produce a compressed production bundle
RUN python train.py && \
    python test2_model_compression.py || echo "Compression step failed — continuing without compressed bundle"

EXPOSE 8000

# 2 Gunicorn workers, 120s timeout (HDBSCAN approximate_predict
# is fast but the first request may need slightly longer)
CMD ["gunicorn", "app:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
