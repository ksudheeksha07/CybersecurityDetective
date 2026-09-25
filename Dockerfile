FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/processed ./data/processed
COPY data/cybersecurity.db ./data/cybersecurity.db

EXPOSE 5000

CMD ["python", "src/api.py"]