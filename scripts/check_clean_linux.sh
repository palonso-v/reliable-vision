#!/bin/sh
# Run inside the pinned CPU serving image with the workspace mounted at /workspace.
set -eu
python -m pip install --no-build-isolation -c requirements-lock.txt -e '.[serve,dev,validation]'
python -m pip check
python -m pytest -q --basetemp=artifacts/final-linux-tests -o cache_dir=artifacts/final-linux-pytest-cache
python -m reliablevision.cli --config configs/smoke.toml --output artifacts/final-linux-smoke --tracking-uri sqlite:///artifacts/final-linux-mlflow.db
python scripts/build_drift_reference.py --artifact artifacts/final-linux-smoke/model.pt
