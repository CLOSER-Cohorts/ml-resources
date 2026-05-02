FROM python:3.11-slim

# Install system dependency needed by scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and model
COPY projects/ ./projects
COPY src/ ./src
COPY config/ ./config
COPY logs/ ./logs
COPY scripts/ .scripts
COPY grafana/ .grafana
#COPY models/ ./models/

# Configuration defaults
#ENV MODEL_PATH=/app/models/classifier_v1.2.pkl
#ENV VECTORISER_PATH=/app/models/tfidf_vectoriser.pkl
ENV PORT=8000
ENV LOG_LEVEL=INFO
ENV PYTHONPATH=/

# Expose the port the server listens on
EXPOSE 8000

# Health check: verify the server is responding
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "projects.am1_project.api.main:app", "--host", "0.0.0.0", "--port", "8000"]