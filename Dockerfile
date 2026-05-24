FROM python:3.12-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p api/models api/data/uploads

EXPOSE 8001

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
