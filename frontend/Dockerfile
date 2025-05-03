# Use a minimal Python base image
FROM python:3.10-slim

# Set the working directory
# WORKDIR /app

# Install required system libraries (e.g., for image processing, XML handling, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to run your app
CMD ["python3", "app.py"]