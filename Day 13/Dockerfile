# ── YOLO Vision · Flask App ────────────────────────────
FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create runtime directories
RUN mkdir -p static/uploads static/results

# Expose Flask port
EXPOSE 5000

# Run with gunicorn in production; use flask dev server for local
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
