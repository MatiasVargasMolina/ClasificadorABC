FROM python:3.9-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /worker

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    curl \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY autosklearn_worker/requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install setuptools==68.2.2 wheel==0.41.3
RUN pip install -r requirements.txt

COPY autosklearn_worker/main.py .
COPY app ./app

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]