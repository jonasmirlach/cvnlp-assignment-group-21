FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

VOLUME ["/app"]

ENV PYTHONUNBUFFERED=1
