# Start from an official Python 3.12 image — this is like a clean machine
# with Python 3.12 already installed, nothing else
FROM python:3.12-slim

# Set the working directory inside the container — all subsequent commands
# run from here, equivalent to cd /app
WORKDIR /app

# Copy requirements.txt first, before copying the rest of the code.
# Docker caches each step — if requirements.txt hasn't changed, it won't
# reinstall packages on every rebuild, making builds much faster
COPY requirements.txt .

# Install all Python packages — same as pip install -r requirements.txt locally
# --no-cache-dir keeps the image smaller by not storing pip's download cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our code into the container
COPY . .

# Run ingest.py to build the ChromaDB vector database from our knowledge files.
# This happens at BUILD time, not at runtime — the database gets baked into
# the container image itself, so the server starts up with it already ready
RUN python app/rag/ingest.py

# Tell Docker that our app listens on port 8080 — Cloud Run specifically
# expects port 8080, unlike our local setup which used 8000
EXPOSE 8080

# Start the FastAPI server when the container runs.
# We use 0.0.0.0 so it accepts connections from outside the container
# (not just localhost), and port 8080 to match Cloud Run's expectation.
# No --reload flag here — that's only for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]