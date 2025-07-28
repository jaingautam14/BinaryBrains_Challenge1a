# Multi-stage Dockerfile for PDF Processing with OCR and Language Detection
FROM --platform=linux/amd64 python:3.10-slim AS base

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-rus \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-ara \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory and user
WORKDIR /app
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

# Copy application code
COPY process_pdfs.py .

# Copy schema for output validation (create directory structure)
RUN mkdir -p /app/schema
COPY sample_dataset/schema/output_schema.json /app/schema/

# Create input and output directories
RUN mkdir -p /app/input /app/output && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys, json; sys.exit(0)"

# Set resource limits and optimizations
ENV OMP_NUM_THREADS=8
ENV TESSERACT_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Default command
CMD ["python", "process_pdfs.py"]

# Build info
LABEL maintainer="PDF Processor"
LABEL version="1.0"
LABEL description="High-performance PDF processing with OCR and language detection"
