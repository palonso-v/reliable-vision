"""Fixed-reference monitoring; no pixels, names, addresses or free-text review notes."""

import hashlib
import json
import math
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

FEATURES = ("brightness_mean", "brightness_std")
EDGES = np.linspace(0, 1, 11)


def input_features(image: torch.Tensor) -> list[float]:
    gray = (image * image.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]).sum(0)
    return [float(gray.mean()), float(gray.std(unbiased=False))]


def reference_document(
    values: list[list[float]], model_version: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    if len(values) < 8:
        raise ValueError("Need at least eight nominal reference images")
    array = np.asarray(values)
    if (
        array.shape != (len(values), 2)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
    ):
        raise ValueError("Invalid reference features")
    return {
        "schema": 1,
        "model_version": model_version,
        "features": list(FEATURES),
        "edges": EDGES.tolist(),
        "count": len(values),
        "provenance": provenance,
        "counts": [np.histogram(array[:, i], EDGES)[0].tolist() for i in range(2)],
    }


class Monitor:
    def __init__(
        self,
        database: Path,
        reference: Path,
        model_version: str,
        window: int = 20,
        threshold: float = 0.3,
        retention_days: int = 30,
    ) -> None:
        raw = reference.read_bytes()
        document = json.loads(raw)
        if (
            document["schema"] != 1
            or document["model_version"] != model_version
            or document["features"] != list(FEATURES)
            or document["edges"] != EDGES.tolist()
            or document["count"] < 8
        ):
            raise ValueError("Reference schema, features or model identity mismatch")
        counts = np.asarray(document["counts"], dtype=float)
        if (
            counts.shape != (2, 10)
            or not np.isfinite(counts).all()
            or (counts < 0).any()
            or not np.all(counts.sum(1) == document["count"])
        ):
            raise ValueError("Invalid reference counts")
        if window < 8 or not 0 < threshold <= 1 or retention_days < 1:
            raise ValueError("Invalid monitoring settings")
        self.reference_id = hashlib.sha256(raw).hexdigest()
        self.model_version = model_version
        self.reference = counts / counts.sum(1, keepdims=True)
        self.window = window
        self.threshold = threshold
        self.database = database
        self.retention_days = retention_days
        self.pending: dict[str, list[list[float]]] = defaultdict(list)
        database.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS drift_windows (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT NOT NULL,
                    reference_id TEXT NOT NULL, traffic TEXT NOT NULL,
                    sample_count INTEGER NOT NULL,
                    statistics TEXT NOT NULL, threshold REAL NOT NULL, flagged INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, request_id TEXT UNIQUE NOT NULL,
                    model_version TEXT NOT NULL, reference_id TEXT NOT NULL, traffic TEXT NOT NULL,
                    score REAL NOT NULL, threshold REAL NOT NULL, reasons TEXT NOT NULL,
                    drift_window_id TEXT, status TEXT NOT NULL DEFAULT 'pending',
                    decision TEXT, resolved_at TEXT);
                CREATE INDEX IF NOT EXISTS review_status_time ON reviews(status,created_at);
                CREATE INDEX IF NOT EXISTS review_retention ON reviews(created_at);
                CREATE INDEX IF NOT EXISTS drift_retention ON drift_windows(created_at);
                CREATE INDEX IF NOT EXISTS drift_identity_time ON
                    drift_windows(model_version,reference_id,traffic,created_at);
            """)
        self.prune()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def prune(self) -> None:
        with self.connect() as connection:
            for table in ("reviews", "drift_windows"):
                connection.execute(
                    f"DELETE FROM {table} WHERE julianday(created_at) < julianday('now') - ?",
                    (self.retention_days,),
                )

    def observe(
        self,
        values: list[float],
        score: float,
        prediction_threshold: float,
        request_id: str,
        traffic: str,
    ) -> dict[str, Any]:
        if (
            traffic not in {"unspecified", "simulated"}
            or len(values) != 2
            or not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            or not math.isfinite(score)
            or not math.isfinite(prediction_threshold)
        ):
            raise ValueError("Invalid observation")
        self.prune()
        buffered = [*self.pending[traffic], values]
        now = datetime.now(UTC).isoformat()
        window_id = None
        distances = None
        flagged = False
        if len(buffered) == self.window:
            array = np.asarray(buffered)
            observed = np.array([np.histogram(array[:, i], EDGES)[0] for i in range(2)])
            distances = (0.5 * np.abs(observed / len(buffered) - self.reference).sum(1)).tolist()
            flagged = max(distances) > self.threshold
            window_id = uuid.uuid4().hex
        reasons = []
        if score > prediction_threshold:
            reasons.append("anomaly_score")
        if flagged:
            reasons.append("input_drift_window")
        review_id = uuid.uuid4().hex if reasons else None
        # Commit window and review together. Advance buffer only after durable success.
        with self.connect() as connection:
            if window_id:
                connection.execute(
                    "INSERT INTO drift_windows VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        window_id,
                        now,
                        self.model_version,
                        self.reference_id,
                        traffic,
                        len(buffered),
                        json.dumps(dict(zip(FEATURES, distances, strict=True))),
                        self.threshold,
                        int(flagged),
                    ),
                )
            if review_id:
                connection.execute(
                    "INSERT INTO reviews (id,created_at,request_id,model_version,"
                    "reference_id,traffic,score,threshold,reasons,drift_window_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        review_id,
                        now,
                        request_id,
                        self.model_version,
                        self.reference_id,
                        traffic,
                        score,
                        prediction_threshold,
                        json.dumps(reasons),
                        window_id,
                    ),
                )
        self.pending[traffic] = [] if window_id else buffered
        return {
            "enabled": True,
            "reference_id": self.reference_id,
            "window_id": window_id,
            "window_complete": window_id is not None,
            "input_drift": flagged if window_id else None,
            "distances": dict(zip(FEATURES, distances, strict=True)) if distances else None,
            "review_id": review_id,
            "traffic": traffic,
        }

    def list_reviews(self, status: str, limit: int) -> list[dict[str, Any]]:
        self.prune()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reviews WHERE status=? ORDER BY created_at,id LIMIT ?",
                (status, limit),
            ).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["reasons"] = json.loads(row["reasons"])
        return result

    def latest_windows(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = [
                connection.execute(
                    "SELECT * FROM drift_windows WHERE model_version=? "
                    "AND reference_id=? AND traffic=? ORDER BY created_at DESC,id "
                    "DESC LIMIT 1",
                    (self.model_version, self.reference_id, traffic),
                ).fetchone()
                for traffic in ("simulated", "unspecified")
            ]
        return [dict(row) for row in rows if row is not None]

    def resolve(self, review_id: str, decision: str) -> str:
        if decision not in {"confirmed_anomaly", "false_alarm", "insufficient_evidence"}:
            raise ValueError("Invalid review decision")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status FROM reviews WHERE id=?", (review_id,)
            ).fetchone()
            if row is None:
                return "missing"
            changed = connection.execute(
                "UPDATE reviews SET status='resolved',decision=?,"
                "resolved_at=? WHERE id=? AND status='pending'",
                (decision, datetime.now(UTC).isoformat(), review_id),
            ).rowcount
        return "resolved" if changed else "conflict"
