FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/PageIndex

EXPOSE 9011

CMD ["python", "-m", "PageIndex.__main__", "--http", "--host", "0.0.0.0", "--port", "9011"]
