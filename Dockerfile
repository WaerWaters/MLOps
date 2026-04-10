FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

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

ARG COMMIT_HASH=unknown
ENV COMMIT_HASH=${COMMIT_HASH}

CMD "python3", "main.py", "${COMMIT_HASH}"
