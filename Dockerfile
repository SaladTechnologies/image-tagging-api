FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
  build-essential \
  git

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -c "from clip import clip; clip.load('ViT-B/16')"

WORKDIR /app/data
COPY src/data .

WORKDIR /app
COPY src/*.py .

CMD ["python", "server.py"]
