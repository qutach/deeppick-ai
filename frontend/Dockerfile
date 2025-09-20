FROM zauberzeug/nicegui:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py /app/
COPY *.js /app/