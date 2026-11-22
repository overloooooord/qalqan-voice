FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QALQAN_HOST=0.0.0.0 \
    QALQAN_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY qalqan ./qalqan
COPY scripts ./scripts
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "qalqan.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
