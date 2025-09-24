FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip and install deps from requirements.txt
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy your app code
COPY . /app

# Start your strategy (change if your entry file is different)
CMD ["python", "Ranked_ML.py"]
