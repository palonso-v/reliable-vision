"""Verify actual local scrape, alert, provisioned dashboard and review workflow."""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-restart", action="store_true")
    parser.add_argument("--queries-only", action="store_true")
    args = parser.parse_args()
    path = Path("artifacts/monitoring-check.json")
    api = "http://127.0.0.1:8000"
    if args.queries_only:
        dashboard = get("http://127.0.0.1:3000/api/dashboards/uid/reliablevision")
        count = 0
        for panel in dashboard["dashboard"]["panels"]:
            for target in panel["targets"]:
                query = urllib.parse.urlencode({"query": target["expr"]})
                assert get("http://127.0.0.1:9090/api/v1/query?" + query)["status"] == "success"
                count += 1
        report = json.loads(path.read_text())
        report["dashboard_queries_validated"] = count
        path.write_text(json.dumps(report, indent=2))
        print(f"All {count} provisioned panel queries accepted by Prometheus")
        return
    if args.after_restart:
        report = json.loads(path.read_text())
        resolved = get(api + "/reviews?status=resolved&limit=100")["items"]
        assert any(
            r["id"] == report["review_id"]
            and r["decision"] == "insufficient_evidence"
            and r["model_version"] == report["model_version"]
            for r in resolved
        )
        report["review_survived_restart"] = True
        path.write_text(json.dumps(report, indent=2))
        print("Review decision and model trace survived API restart")
        return
    query = urllib.parse.urlencode({"query": 'up{job="reliablevision"}'})
    scrape = get("http://127.0.0.1:9090/api/v1/query?" + query)
    assert scrape["data"]["result"][0]["value"][1] == "1"
    alerts = []
    for _ in range(12):
        alerts = get("http://127.0.0.1:9090/api/v1/alerts")["data"]["alerts"]
        if any(
            a["labels"].get("alertname") == "ReliableVisionInputDrift"
            and a["labels"].get("traffic") == "simulated"
            and a["state"] == "firing"
            for a in alerts
        ):
            break
        time.sleep(5)
    else:
        raise RuntimeError("Simulated input drift alert did not fire")
    assert get("http://127.0.0.1:3000/api/health")["database"] == "ok"
    dashboard = get("http://127.0.0.1:3000/api/dashboards/uid/reliablevision")
    assert len(dashboard["dashboard"]["panels"]) == 8
    items = get(api + "/reviews?limit=100")["items"]
    selected = next(r for r in items if r["traffic"] == "simulated")
    assert not {"image", "anomaly_map", "notes", "ip", "filename"} & set(selected)
    request = urllib.request.Request(
        api + "/reviews/" + selected["id"],
        method="PATCH",
        data=json.dumps({"decision": "insufficient_evidence"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        assert json.load(response)["status"] == "resolved"
    report = {
        "traffic": "simulated",
        "scrape_up": True,
        "drift_alert_firing": True,
        "dashboard_panels": 8,
        "pending_reviews_before_resolution": len(items),
        "review_id": selected["id"],
        "model_version": selected["model_version"],
        "reference_id": selected["reference_id"],
        "decision": "insufficient_evidence",
        "review_survived_restart": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
