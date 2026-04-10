FROM python:3.10-slim

WORKDIR /app

RUN apt-get clean && apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copy and install requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code
COPY api/ ./api/
COPY img2img-turbo/src/ ./src/

# expose port
EXPOSE 8000

# run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]