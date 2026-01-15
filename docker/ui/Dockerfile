FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ui.txt /app/
RUN pip install -r requirements-ui.txt

COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

