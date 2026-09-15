FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic

# Installs dev extras (pytest, etc.) too: this image is used for local
# development via docker-compose (bind-mounted source, --reload below), not
# as a slim production image, so having the dev tooling available inside
# the container is worth the extra size.
RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000

# --reload relies on the docker-compose bind mount of ./backend:/app so
# code edits on the host are picked up without rebuilding the image.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
