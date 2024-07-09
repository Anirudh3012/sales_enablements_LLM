FROM python:3.10-slim

WORKDIR /app

# Install necessary tools for building some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libice-dev \
    libsm-dev
CMD ["pip", "install" ,"--upgrade", "numpy" ,"scipy"]

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


CMD ["uvicorn", "main:fastapi_app", "--host", "0.0.0.0", "--port", "4000", "--reload"]
