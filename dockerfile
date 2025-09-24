FROM python:3.11-slim

# Keep Python output unbuffered and pip lean
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy your app code
COPY . /app

# Start your strategy
CMD ["python", "Ranked_ML.py"]
