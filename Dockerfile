FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 libpng16-16 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p database data/uploads data/working outputs logs memory

# Set environment variables
ENV STATIS_HOST=0.0.0.0
ENV STATIS_PORT=8420
ENV STATIS_DEBUG=false

# Expose port
EXPOSE 8420

# Run the application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8420"]
