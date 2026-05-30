# Base Python Image
FROM python:3.11-slim

# Set Working Directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements File
COPY requirements.txt .

# Install Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Full Project
COPY . .

# Streamlit Port
EXPOSE 8501

# Run Streamlit App
CMD ["streamlit", "run", "creditscoringapp.py", "--server.port=8501", "--server.address=0.0.0.0"]