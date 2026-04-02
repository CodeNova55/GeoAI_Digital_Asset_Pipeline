# GeoAI Digital Asset Pipeline - Docker Configuration
# =====================================================

FROM osgeo/gdal:ubuntu-full-3.8.0

LABEL maintainer="GeoAI Research Team"
LABEL description="GeoAI Digital Asset Pipeline - QGIS + AI integration for geospatial analysis"
LABEL version="1.0.0"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package
RUN pip3 install -e .

# Create directories
RUN mkdir -p /app/data/raw /app/data/processed /app/outputs /app/models /app/logs

# Set permissions
RUN chmod -R 755 /app

# Default command
CMD ["python", "-m", "scripts.cli", "--help"]

# Expose port for API (if using FastAPI)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"
