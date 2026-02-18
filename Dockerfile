FROM python:3.11

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

RUN python3 pytest

