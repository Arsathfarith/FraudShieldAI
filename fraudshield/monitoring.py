import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_runtime_override = os.environ.get("FRAUDSHIELD_RUNTIME_DIR")
RUNTIME_ROOT = Path(_runtime_override) if _runtime_override else (Path(tempfile.gettempdir()) / "fraudshield-ai" if os.environ.get("VERCEL") else PROJECT_ROOT)
LOG_DIR = RUNTIME_ROOT / "logs"
PREDICTION_LOG = LOG_DIR / "prediction_logs.csv"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api_status():
    return {
        "status": "online",
        "service": "FraudShield AI",
        "checked_at": utc_now_iso(),
    }


def log_prediction(result):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "prediction", "label", "confidence", "fraud_probability"]
    new_file = not PREDICTION_LOG.exists()
    with PREDICTION_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": utc_now_iso(),
            "prediction": result["prediction"],
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "fraud_probability": round(result["fraud_probability"], 4),
        })


def read_prediction_logs(limit=50):
    if not PREDICTION_LOG.exists():
        return []
    with PREDICTION_LOG.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-limit:][::-1]


def prediction_summary(logs):
    total = len(logs)
    fraud = 0
    legitimate = 0
    probabilities = []

    for row in logs:
        label = row.get("label", "").lower()
        prediction = str(row.get("prediction", ""))
        if "fraud" in label or prediction == "1":
            fraud += 1
        else:
            legitimate += 1

        try:
            probabilities.append(float(row.get("fraud_probability", 0)))
        except (TypeError, ValueError):
            probabilities.append(0.0)

    average_probability = sum(probabilities) / total if total else 0.0
    latest_probability = probabilities[0] if probabilities else 0.0
    fraud_rate = fraud / total if total else 0.0

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": fraud_rate,
        "average_probability": average_probability,
        "latest_probability": latest_probability,
        "chart": [
            {"label": "Fraud", "value": fraud},
            {"label": "Legitimate", "value": legitimate},
        ],
    }
