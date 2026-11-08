"""Prepare only workspace Docker-state directories with container UID ownership.

Run in the built CPU image as root with the workspace mounted, as documented.
On Windows, creation alone suffices under Docker Desktop's filesystem mapping.
"""

import os
from pathlib import Path


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    review = Path(os.environ.get("RV_STATE_DIR", str(workspace / "artifacts/monitoring")))
    if not review.is_absolute():
        review = workspace / review
    paths = [
        (review.resolve(), 10001, 10001),
        (workspace / "artifacts/prometheus", 65534, 65534),
        (workspace / "artifacts/grafana", 472, 0),
    ]
    for path, uid, gid in paths:
        if not path.resolve().is_relative_to(workspace):
            raise ValueError("State directories must stay in the workspace")
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            if os.geteuid() != 0:
                raise PermissionError("Run the documented Docker initialization as root")
            os.chown(path, uid, gid)
            path.chmod(0o750)
        print(f"Prepared state directory: {path}")


if __name__ == "__main__":
    main()
