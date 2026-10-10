FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROJECT_ROOT=/workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy scripts into container
COPY Scripts /workspace/Scripts

# Copy models into container
COPY Models /workspace/Models

# Install Python requirements
COPY Utils/requirements.txt /tmp/project-requirements.txt
COPY TomoDockerPipeline/requirements-ui.txt /tmp/requirements-ui.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /tmp/project-requirements.txt && \
    python -m pip install -r /tmp/requirements-ui.txt

# Copy UI
COPY TomoDockerPipeline/app.py /app/app.py
COPY TomoDockerPipeline/templates /app/templates

EXPOSE 8080

CMD ["python","/app/app.py"]


