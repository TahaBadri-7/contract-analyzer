# The recipe for the box that runs the service. A host (Render, in part 3) builds this
# file into an image and runs it; it never needs to know anything about Python.

# A small Linux with Python already installed. 3.13 rather than the newest Python:
# mature versions have ready-made packages for everything, so the build never compiles.
FROM python:3.13-slim

# Print logs immediately (useful on a host's log page) and skip writing .pyc files.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install the packages first, in their own step. Docker reuses this step on the next build
# as long as requirements.txt has not changed, so code edits rebuild in seconds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service needs. The evals, the dataset and the results stay out.
COPY extractor.py retriever.py documents.py api.py ./
COPY static ./static

# Run as an ordinary user, not root: if anything ever ran inside this box that should not,
# it could not change the system. (burnbox's image did the same.)
RUN useradd --create-home appuser
USER appuser

# Hosts tell the app which port to listen on through $PORT; 8000 when running locally.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]