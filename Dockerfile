# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Set work directory
WORKDIR /app

# Install system dependencies
# gcc and python3-dev are often needed for building python packages like numpy/pandas/chromadb
# default-libmysqlclient-dev is needed for pymysql/mysqlclient if used (though pymysql is pure python, sometimes dependencies need headers)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs outputs/plots brain

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
