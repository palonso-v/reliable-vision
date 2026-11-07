"""Run the unmodified training CLI and sample its process RSS (native allocations included)."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter, sleep

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mvtec.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bottle-profile.json"))
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path.cwd().resolve()):
        parser.error("Output must stay inside workspace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    peak = 0
    os_peak = 0
    samples = []
    with args.output.with_suffix(".log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "reliablevision.cli", "--config", str(args.config)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        process = psutil.Process(child.pid)
        last_report = -30.0
        while child.poll() is None:
            try:
                # Windows venv executables can be launchers with an interpreter child.
                members = [process, *process.children(recursive=True)]
                memories = []
                for member in members:
                    try:
                        memories.append(member.memory_info())
                    except psutil.NoSuchProcess:
                        continue
                rss = sum(memory.rss for memory in memories)
                peak = max(peak, rss)
                os_peak = max([os_peak, *(getattr(m, "peak_wset", 0) for m in memories)])
                seconds = perf_counter() - started
                samples.append({"seconds": seconds, "rss_bytes": rss})
                if seconds - last_report >= 30:
                    print(
                        f"Run active: {seconds:.0f}s, RSS {rss / 2**20:.0f} MiB, "
                        f"peak {peak / 2**20:.0f} MiB",
                        flush=True,
                    )
                    last_report = seconds
            except psutil.NoSuchProcess:
                break
            sleep(0.25)
        code = child.wait()
    report = {
        "command": ["python", "-m", "reliablevision.cli", "--config", str(args.config)],
        "exit_code": code,
        "elapsed_seconds": perf_counter() - started,
        "peak_rss_bytes": peak,
        "max_individual_os_peak_rss_bytes": os_peak,
        "sampling_interval_seconds": 0.25,
        "memory_scope": "Sampled process-tree RSS; includes Windows launcher and interpreter",
        "samples": samples,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if code:
        print(args.output.with_suffix(".log").read_text(encoding="utf-8")[-6000:])
        raise SystemExit(code)
    print(f"Run completed; resource evidence: {args.output}")


if __name__ == "__main__":
    main()
