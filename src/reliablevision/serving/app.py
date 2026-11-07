import json
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from reliablevision.serving.runtime import APIError, Runtime
from reliablevision.serving.settings import Settings


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str
    artifact_schema: int
    algorithm: str
    category: str
    data_mode: str
    synthetic: bool
    device: str
    image_size: int
    threshold: float
    score_semantics: str
    registry_name: str | None
    registry_version: int | str | None
    torch_threads: int
    max_concurrent_predictions: int


class Prediction(ModelInfo):
    request_id: str
    score: float
    is_anomaly: bool
    anomaly_map: list[list[float]]
    map_shape: list[int]
    map_coordinates: str
    inference_seconds: float


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "timestamp": datetime.now(UTC).isoformat(),
                "event": record.getMessage(),
                **getattr(record, "fields", {}),
            },
            allow_nan=False,
        )


logger = logging.getLogger("reliablevision.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    headers = {"Retry-After": "1"} if code == "busy" else None
    return JSONResponse(
        status_code=status,
        headers=headers,
        content={
            "error": {"code": code, "message": message},
            "request_id": request.state.request_id,
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    registry = CollectorRegistry()
    requests = Counter(
        "rv_http_requests_total",
        "Completed API requests",
        ["route", "method", "status"],
        registry=registry,
    )
    latency = Histogram(
        "rv_http_request_seconds",
        "Request time including serialization",
        ["route"],
        registry=registry,
    )
    inference = Histogram(
        "rv_inference_seconds",
        "Model forward time only",
        registry=registry,
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
    )
    scores = Histogram(
        "rv_anomaly_score",
        "Raw distance distribution, not probabilities",
        registry=registry,
        buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100),
    )
    active = Gauge("rv_predictions_active", "Admitted prediction requests", registry=registry)
    admission = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = Runtime(settings or Settings.from_env())
        logger.info("model_loaded", extra={"fields": app.state.runtime.info})
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="ReliableVision", version="0.2.0", lifespan=lifespan)

    @app.middleware("http")
    async def instrument(request: Request, call_next: Any) -> Response:
        request.state.request_id = uuid.uuid4().hex
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            logger.error(
                "request_failed",
                extra={
                    "fields": {
                        "request_id": request.state.request_id,
                        "error_code": "internal_error",
                        "error_type": type(error).__name__,
                    }
                },
            )
            response = error_response(request, 500, "internal_error", "Internal inference error")
        duration = perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "other")
        if route not in {"/predict", "/health", "/model-info", "/metrics"}:
            route = "other"
        method = request.method if request.method in {"GET", "POST", "HEAD", "OPTIONS"} else "other"
        requests.labels(route, method, str(response.status_code)).inc()
        latency.labels(route).observe(duration)
        logger.info(
            "request_completed",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "route": route,
                    "method": method,
                    "status": response.status_code,
                    "duration_seconds": duration,
                }
            },
        )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(APIError)
    async def api_error(request: Request, error: APIError) -> JSONResponse:
        return error_response(request, error.status, error.code, error.message)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        response = error_response(request, error.status_code, "http_error", "Request not supported")
        response.headers.update(error.headers or {})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return error_response(request, 422, "invalid_request", "Request validation failed")

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            raise APIError(503, "not_ready", "Model is not ready")
        return {"status": "ready", "model_version": runtime.info["model_version"]}

    @app.get("/model-info", response_model=ModelInfo)
    async def model_info(request: Request) -> dict[str, Any]:
        return request.app.state.runtime.info

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post(
        "/predict",
        response_model=Prediction,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    mime: {"schema": {"type": "string", "format": "binary"}}
                    for mime in ("image/png", "image/jpeg")
                },
            }
        },
    )
    async def predict(request: Request) -> dict[str, Any]:
        runtime = request.app.state.runtime
        if not admission.acquire(blocking=False):
            raise APIError(503, "busy", "Prediction capacity is occupied; retry later")
        active.inc()
        try:
            media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if media_type not in {"image/png", "image/jpeg"}:
                raise APIError(415, "unsupported_media_type", "Send a PNG or JPEG image body")
            limit = runtime.settings.max_upload_bytes
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as error:
                    raise APIError(
                        400, "invalid_content_length", "Invalid Content-Length"
                    ) from error
                if declared < 0:
                    raise APIError(400, "invalid_content_length", "Invalid Content-Length")
                if declared > limit:
                    raise APIError(413, "body_too_large", "Image body exceeds byte limit")
            body = bytearray()
            try:
                async for chunk in request.stream():
                    if len(body) + len(chunk) > limit:
                        raise APIError(413, "body_too_large", "Image body exceeds byte limit")
                    body.extend(chunk)
            except ClientDisconnect as error:
                raise APIError(400, "client_disconnected", "Incomplete request body") from error
            if not body:
                raise APIError(422, "empty_image", "Image body is empty")
            result = await anyio.to_thread.run_sync(runtime.predict, bytes(body), media_type)
            inference.observe(result["inference_seconds"])
            scores.observe(result["score"])
            return {"request_id": request.state.request_id, **result}
        finally:
            active.dec()
            admission.release()

    return app
