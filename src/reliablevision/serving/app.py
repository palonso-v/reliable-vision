import json
import logging
import sqlite3
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import anyio
from fastapi import FastAPI, Query, Request
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

from reliablevision.monitoring import FEATURES, Monitor
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
    monitoring: dict[str, Any]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["confirmed_anomaly", "false_alarm", "insufficient_evidence"]


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
        ["route", "method", "status", "traffic"],
        registry=registry,
    )
    latency = Histogram(
        "rv_http_request_seconds",
        "Request time including serialization",
        ["route", "traffic"],
        registry=registry,
    )
    inference = Histogram(
        "rv_inference_seconds",
        "Model forward time only",
        ["traffic"],
        registry=registry,
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
    )
    scores = Histogram(
        "rv_anomaly_score",
        "Raw distance distribution, not probabilities",
        ["traffic"],
        registry=registry,
        buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 100),
    )
    active = Gauge("rv_predictions_active", "Admitted prediction requests", registry=registry)
    admission = threading.Lock()
    model_identity = Gauge(
        "rv_model_info",
        "Loaded artifact identity",
        ["model_version", "data_mode"],
        registry=registry,
    )
    errors = Counter(
        "rv_http_errors_total", "HTTP errors", ["route", "status", "traffic"], registry=registry
    )
    drift_distance = Gauge(
        "rv_input_drift_tv",
        "Last completed window total variation distance",
        ["feature", "traffic"],
        registry=registry,
    )
    drift_flag = Gauge(
        "rv_input_drift_flag",
        "Last completed window flagged (not performance loss)",
        ["traffic"],
        registry=registry,
    )
    windows = Counter(
        "rv_drift_windows_total",
        "Completed fixed reference windows",
        ["traffic", "flagged"],
        registry=registry,
    )
    monitoring_failures = Counter(
        "rv_monitoring_failures_total", "Monitoring storage failures", registry=registry
    )
    drift_updated = Gauge(
        "rv_input_drift_last_window_timestamp_seconds",
        "Window freshness",
        ["traffic"],
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = Runtime(settings or Settings.from_env())
        chosen = app.state.runtime.settings
        app.state.monitor = (
            Monitor(
                Path(chosen.database),
                Path(chosen.reference),
                app.state.runtime.info["model_version"],
                chosen.drift_window,
                chosen.drift_threshold,
                chosen.retention_days,
            )
            if chosen.monitoring
            else None
        )
        model_identity.labels(
            app.state.runtime.info["model_version"], app.state.runtime.info["data_mode"]
        ).set(1)
        if app.state.monitor is not None:
            for window in app.state.monitor.latest_windows():
                traffic = window["traffic"]
                drift_flag.labels(traffic).set(window["flagged"])
                for feature, value in json.loads(window["statistics"]).items():
                    drift_distance.labels(feature, traffic).set(value)
                drift_updated.labels(traffic).set(
                    datetime.fromisoformat(window["created_at"]).timestamp()
                )
        logger.info("model_loaded", extra={"fields": app.state.runtime.info})
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="ReliableVision", version="0.3.0", lifespan=lifespan)

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
        if route not in {
            "/predict",
            "/health",
            "/model-info",
            "/metrics",
            "/reviews",
            "/reviews/{review_id}",
        }:
            route = "other"
        method = request.method if request.method in {"GET", "POST", "HEAD", "OPTIONS"} else "other"
        traffic = "control"
        if route == "/predict":
            traffic = request.headers.get("x-rv-traffic", "unspecified")
            if traffic not in {"simulated", "unspecified"}:
                traffic = "unspecified"
        requests.labels(route, method, str(response.status_code), traffic).inc()
        if response.status_code >= 400:
            errors.labels(route, str(response.status_code), traffic).inc()
        latency.labels(route, traffic).observe(duration)
        logger.info(
            "request_completed",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "route": route,
                    "method": method,
                    "status": response.status_code,
                    "duration_seconds": duration,
                    "traffic": traffic,
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

    def require_monitor(request: Request) -> Monitor:
        if request.app.state.monitor is None:
            raise APIError(503, "monitoring_disabled", "Review monitoring is not enabled")
        return request.app.state.monitor

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, error: sqlite3.Error) -> JSONResponse:
        monitoring_failures.inc()
        return error_response(
            request, 503, "monitoring_unavailable", "Monitoring storage unavailable"
        )

    @app.get("/reviews")
    def reviews(
        request: Request,
        status: Literal["pending", "resolved"] = "pending",
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, Any]:
        return {"items": require_monitor(request).list_reviews(status, limit)}

    @app.patch("/reviews/{review_id}")
    def resolve_review(
        request: Request, review_id: uuid.UUID, body: ReviewDecision
    ) -> dict[str, str]:
        result = require_monitor(request).resolve(review_id.hex, body.decision)
        if result == "missing":
            raise APIError(404, "review_not_found", "Review record does not exist")
        if result == "conflict":
            raise APIError(409, "review_already_resolved", "Review decision is immutable")
        return {"id": review_id.hex, "status": result, "decision": body.decision}

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
            features = result.pop("_input_features")
            traffic = request.headers.get("x-rv-traffic", "unspecified")
            if traffic not in {"unspecified", "simulated"}:
                raise APIError(
                    422, "invalid_traffic_label", "Traffic must be unspecified or simulated"
                )
            monitor = request.app.state.monitor
            observation = {"enabled": False, "traffic": traffic}
            if monitor is not None:
                observation = await anyio.to_thread.run_sync(
                    monitor.observe,
                    features,
                    result["score"],
                    result["threshold"],
                    request.state.request_id,
                    traffic,
                )
                if observation["window_complete"]:
                    for feature in FEATURES:
                        drift_distance.labels(feature, traffic).set(
                            observation["distances"][feature]
                        )
                    drift_flag.labels(traffic).set(int(observation["input_drift"]))
                    windows.labels(traffic, str(observation["input_drift"]).lower()).inc()
                    drift_updated.labels(traffic).set(datetime.now(UTC).timestamp())
            inference.labels(traffic).observe(result["inference_seconds"])
            scores.labels(traffic).observe(result["score"])
            return {"request_id": request.state.request_id, "monitoring": observation, **result}
        finally:
            active.dec()
            admission.release()

    return app
