FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Configure pip with longer timeouts and retries
ENV PIP_TIMEOUT=1000
ENV PIP_RETRIES=10
ENV PIP_DEFAULT_TIMEOUT=1000

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages with increased timeout and cache enabled
# Split into multiple steps for better error handling and caching
RUN pip install --upgrade pip && \
    pip install --timeout=1000 --retries=10 \
    fastapi==0.115.6 uvicorn[standard]==0.34.0 gunicorn==23.0.0

RUN pip install --timeout=1000 --retries=10 \
    sqlalchemy==2.0.36 pymysql==1.1.1 cryptography==44.0.0

RUN pip install --timeout=1000 --retries=10 \
    python-multipart==0.0.17 python-dotenv==1.0.1

# Install ML packages separately (these are the largest)
RUN pip install --timeout=1000 --retries=10 \
    numpy==1.26.4

RUN pip install --timeout=1000 --retries=10 \
    pandas==2.2.3

RUN pip install --timeout=1000 --retries=10 \
    scikit-learn==1.5.2 joblib==1.4.2

# Copy application code
COPY . .

# Create uploads and models directories
RUN mkdir -p uploads models

# Expose port
EXPOSE 8000

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "main:app"]