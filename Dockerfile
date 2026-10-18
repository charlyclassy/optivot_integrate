# DepotWatt production image.
#   docker build -t depotwatt .
#   docker run -p 8501:8501 -v depotwatt_state:/state depotwatt
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEPOTWATT_ENV=production \
    DEPOTWATT_STATE_DIR=/state

WORKDIR /app

# libgomp is required by XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py config.py data_processing.py evaluation.py feature_engineering.py forecasting.py model_loader.py ./
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY models/global_model ./models/global_model
COPY data/processed ./data/processed
COPY data/sample_processed_data.csv ./data/sample_processed_data.csv
COPY .streamlit ./.streamlit

# Run as an unprivileged user; /state holds the database and customer uploads (mount a volume).
RUN useradd --create-home appuser && mkdir -p /state /app/outputs /app/models/archive \
    && chown -R appuser:appuser /state /app
USER appuser
VOLUME ["/state"]

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
