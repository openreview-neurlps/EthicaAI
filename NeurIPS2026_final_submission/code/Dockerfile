FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Default: run full reproduction pipeline (all 7 experiments)
# For quick smoke test: docker run -e ETHICAAI_FAST=1 ethicaai python scripts/reproduce_fast.py
CMD ["python", "scripts/reproduce_all.py"]
