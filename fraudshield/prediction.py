import pickle

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fraudshield.training import ARTIFACT_PATH, MODEL_PATH


def load_artifacts(required=True):
    if not MODEL_PATH.exists() or not ARTIFACT_PATH.exists():
        if required:
            raise RuntimeError("No trained model found. Upload a dataset and train the project first.")
        return None

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with ARTIFACT_PATH.open("rb") as artifact_file:
        artifacts = pickle.load(artifact_file)
    artifacts["model"] = model
    return artifacts


def coerce_input(values, artifacts):
    row = {}
    for feature in artifacts["feature_columns"]:
        raw_value = values.get(feature, "")
        if feature in artifacts["numeric_features"] or is_numeric_dtype(artifacts.get("feature_dtypes", {}).get(feature)):
            row[feature] = pd.to_numeric(raw_value, errors="coerce")
        else:
            row[feature] = str(raw_value)
    return pd.DataFrame([row], columns=artifacts["feature_columns"])


def predict_transaction(values, artifacts=None):
    artifacts = artifacts or load_artifacts(required=True)
    frame = coerce_input(values, artifacts)
    model = artifacts["model"]
    prediction = int(model.predict(frame)[0])

    probabilities = _prediction_probabilities(model, frame)
    fraud_class = artifacts["fraud_class"]
    fraud_probability = float(probabilities.get(fraud_class, 0.0))
    confidence = max(probabilities.values()) if probabilities else 1.0
    label = artifacts["class_labels"].get(prediction, str(prediction))
    explanations = explain_prediction(frame, artifacts, fraud_probability)

    return {
        "prediction": prediction,
        "label": label,
        "confidence": float(confidence),
        "fraud_probability": fraud_probability,
        "input": values,
        "explanations": explanations,
    }


def _prediction_probabilities(model, frame):
    if not hasattr(model, "predict_proba"):
        return {}
    classes = list(model.classes_)
    probs = model.predict_proba(frame)[0]
    return {int(cls): float(prob) for cls, prob in zip(classes, probs)}


def explain_prediction(frame, artifacts, fraud_probability):
    explanations = []
    importances = artifacts.get("feature_importance", {})
    numeric_stats = artifacts.get("numeric_stats", {})

    for feature, value in frame.iloc[0].items():
        importance = float(importances.get(feature, 0.0))
        reason = "This field contributed context to the model decision."

        if feature in numeric_stats:
            median = numeric_stats[feature].get("median")
            if pd.notna(value) and median is not None:
                difference = float(value) - float(median)
                direction = "above" if difference >= 0 else "below"
                reason = f"Value is {direction} the training median of {median:.2f}."
        elif value:
            reason = "Categorical value was encoded and compared with patterns learned from the dataset."

        score = importance
        if feature in numeric_stats and pd.notna(value):
            spread = numeric_stats[feature].get("iqr") or 1.0
            score += min(abs(float(value) - numeric_stats[feature].get("median", 0.0)) / spread, 3.0) * 0.03

        explanations.append({
            "feature": feature,
            "value": "" if pd.isna(value) else value,
            "importance": round(score, 4),
            "reason": reason,
        })

    explanations.sort(key=lambda item: item["importance"], reverse=True)
    summary = "Higher fraud probability was driven by the strongest feature signals." if fraud_probability >= 0.5 else "The strongest feature signals aligned more closely with legitimate patterns."
    return {"summary": summary, "drivers": explanations[:8]}
