FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# Install the application dependencies
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages dvc dvc-s3
COPY . .
RUN git init && dvc pull --force

# Setup an app user so the container doesn't run as the root user + give ownership
RUN useradd -m app && chown -R app:app /app
USER app

RUN pytest tests

ARG COMMIT_HASH=unknown
ENV COMMIT_HASH=${COMMIT_HASH}

CMD "python3", "main.py", "${COMMIT_HASH}"
