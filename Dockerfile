# Production Dockerfile for Dhaka AQI Route Optimizer + Research Platform
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/

# Expose Django port
EXPOSE 8000

# Default command to run Django production server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
