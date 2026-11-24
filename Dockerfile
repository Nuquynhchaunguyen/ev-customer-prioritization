FROM python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements-service.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-service.txt
RUN groupadd --gid 10001 appuser && useradd --uid 10001 --gid appuser --no-create-home appuser
COPY ev_project/ ./ev_project/
COPY models/model_bundle.joblib models/model_metadata.json models/monitoring_reference.json ./models/
COPY data/demo/customer_profiles_demo.csv data/demo/customer_profiles_template.csv ./data/demo/
COPY tests/test_api.py tests/test_monitoring.py ./tests/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]
CMD ["python", "-m", "uvicorn", "ev_project.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
