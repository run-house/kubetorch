FROM python:3.11-slim

WORKDIR /kt_app

# Install rsync and screen
RUN apt-get update && \
    apt-get install -y rsync screen coreutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python 3.9 using uv
RUN pip install uv
RUN uv python install 3.9
RUN uv venv --seed
RUN uv pip install fastapi jinja2 kubernetes pydantic python-json-logger pyyaml requests uvicorn

# Ensure .venv path is included in image
ENV PATH="$PATH:/kt_app/.venv/bin"

EXPOSE 32300
