# Official CPython 3.13.15, linux/amd64 manifest; deliberately immutable.
FROM python:3.13.15-slim-bookworm@sha256:3e2de9c40ca4e3d73240059f9d48baff27908f10293e985a2f382a0378e6df4a

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 RV_MODEL_PATH=/app/models/model.pt \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY requirements/serve.lock /app/requirements/serve.lock
RUN python -m pip install --no-cache-dir --require-hashes -r requirements/serve.lock \
    && python -m pip check
COPY src/reliablevision /app/src/reliablevision
COPY configs/serve.toml /app/configs/serve.toml
ENV PYTHONPATH=/app/src
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]
CMD ["python", "-m", "uvicorn", "reliablevision.serving.asgi:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--timeout-keep-alive", "5"]
