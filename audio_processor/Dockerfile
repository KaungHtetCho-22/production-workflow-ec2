FROM python:3.9-slim

WORKDIR /app

COPY ../requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# CMD ["python", "audio_process.py"]

# CMD ["bin", "bash"]

# CMD ["python", "sqlmodel.py", "&&", "python", "audio_process.py"]