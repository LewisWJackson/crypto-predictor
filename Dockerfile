# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY web/package*.json ./
RUN npm ci
COPY web/ .
RUN npm run build

# Stage 2: Python backend + serve frontend
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies (CPU-only torch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.0+cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY models/ models/

# Copy built frontend
COPY --from=frontend /app/dist web/dist

EXPOSE 8000

CMD python scripts/serve_api.py \
    --checkpoint models/tft/best.ckpt \
    --config configs/experiments/p1_horizon_15min_agg.yaml \
    --port ${PORT:-8000}
