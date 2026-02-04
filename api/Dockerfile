# -------------------------
# FairCredit API Dockerfile
# -------------------------
FROM python:3.10-slim

# Security: non-root user
RUN useradd -m appuser
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/src api/src
COPY src src

# Switch user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
