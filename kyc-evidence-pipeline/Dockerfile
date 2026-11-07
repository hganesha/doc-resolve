# Offline evidence pipeline behind the authenticated HTTP API in kyc_pipeline/server.py.
# Live mode refuses to start without EVIDENCE_PIPELINE_TOKEN; EVIDENCE_PIPELINE_MODE=demo
# accepts the published demo token and is for synthetic data only.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EVIDENCE_PIPELINE_HOST=0.0.0.0 EVIDENCE_PIPELINE_PORT=8080 \
    EVIDENCE_PIPELINE_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
 && useradd --uid 10001 --create-home pipeline \
 && mkdir -p /data && chown pipeline:pipeline /data && chmod 700 /data
USER pipeline
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"]
CMD ["kyc-evidence-server"]
