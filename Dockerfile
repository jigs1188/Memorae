FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY memorae/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi "uvicorn[standard]" rank-bm25

# Copy the full project
COPY . .

# Expose the API port
EXPOSE 8000

# Set Python path so imports inside memorae/ resolve correctly
ENV PYTHONPATH=/app/memorae

# Change into memorae dir and launch the API
WORKDIR /app/memorae
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
