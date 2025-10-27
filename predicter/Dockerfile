FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ranker.py /app/ranker.py

# models volume is expected at /models via docker-compose
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "/app/ranker.py"]
