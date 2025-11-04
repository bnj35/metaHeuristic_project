FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for pyvrp compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project configuration
COPY app/pyproject.toml ./

# Copy the main algorithm module
COPY app/metaHeuristic.py ./
COPY app/__init__.py ./

# Install the package in development mode
RUN pip install -e .[dev]

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]