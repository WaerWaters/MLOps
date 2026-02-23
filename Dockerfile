FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir dvc dvc-s3
COPY . .
RUN git init && dvc pull --force

# Setup an app user so the container doesn't run as the root user + give ownership
RUN useradd -m app && chown -R app:app /app
USER app

RUN pytest tests

CMD ["python", "main.py"]
