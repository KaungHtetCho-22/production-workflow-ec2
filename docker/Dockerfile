FROM python:3.12 AS prod

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "monsoon_audio_biodiversity/audio_processor/audio_process.py"]
