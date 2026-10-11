   FROM python:3.11-slim
   WORKDIR /app

   COPY requirements-api.txt .
   RUN pip install --no-cache-dir -r requirements-api.txt

   COPY src ./src
   RUN mkdir -p checkpoints

   ENV CHECKPOINT_URL=https://github.com/AyanMoeen/AI-Engineering-Assignment-1/releases/download/v1.0/tiny_gpt.pt

   CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
