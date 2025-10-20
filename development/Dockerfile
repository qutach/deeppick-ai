# syntax=docker/dockerfile:1

FROM python:3.13-slim

# --- Base env (runtime friendly) ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Alembic will read config from this file; adjust if yours is elsewhere
    ALEMBIC_INI=/app/alembic.ini

WORKDIR /app

# --- System deps (keep slim; rely on psycopg2-binary/psycopg[binary]) ---
RUN apt-get update && apt-get install -y locales && \
    sed -i '/sv_SE.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen sv_SE.UTF-8
ENV LANG=sv_SE.UTF-8
ENV LC_ALL=sv_SE.UTF-8

ENV LANG=sv_SE.UTF-8
ENV LC_ALL=sv_SE.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- Install Python deps ---
# If you have a global requirements.txt at repo root (recommended), this uses it.
COPY requirements.txt /app/requirements.txt

# --- Copy full repo so env.py can import your models (target_metadata) ---
# (Make sure .dockerignore excludes venv/, .git/, etc.)
COPY . /app
COPY init_db/init_0.py /app/init_db/init_0.py

CMD ["tail", "-f", "/dev/null"]