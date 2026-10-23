FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY yoloboxer.py yolotiler.py ./
COPY static ./static

# Mount your image directory here; labels are written to /data/labels.
VOLUME /data
EXPOSE 8000 8001

# YOLOboxer by default; run YOLOtiler with
#   python yolotiler.py /data --host 0.0.0.0 --port 8001

CMD ["python", "yoloboxer.py", "/data", "--host", "0.0.0.0", "--port", "8000"]
