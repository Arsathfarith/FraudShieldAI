import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_fraud_model.pkl"
ARTIFACT_PATH = MODEL_DIR / "model_artifacts.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"


TARGET_HINTS = ("fraud", "is_fraud", "class", "label", "target", "status", "outcome", "risk")
DROP_HINTS = ("id", "uuid", "guid", "name", "email", "phone", "address")


def train_from_csv(file_storage):
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    dataset_path = DATA_DIR / "uploaded_dataset.csv"
    file_storage.save(dataset_path)

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    df = _clean_columns(df)
    target_column = _choose_target_column(df)
    feature_columns = _choose_feature_columns(df, target_column)
    if not feature_columns:
        raise ValueError("No usable feature columns were found after dataset analysis.")

    y_raw = df[target_column]
    y, class_labels, fraud_class = _encode_target(y_raw)
    X = _feature_engineering(df[feature_columns])

    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    if y.nunique() < 2:
        raise ValueError("The selected target column has fewer than two classes after cleaning.")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    preprocessor = _build_preprocessor(numeric_features, categorical_features)

    stratify = y if y.value_counts().min() >= 2 else None
    test_size = 0.2 if len(df) >= 50 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    models = _candidate_models(len(X_train), y.nunique())
    results = []
    best_pipeline = None
    best_score = -1.0

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", clone(preprocessor)), ("model", estimator)])
        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            result = {
                "model": name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                "status": "trained",
            }
            results.append(result)
            if result["f1"] > best_score:
                best_score = result["f1"]
                best_pipeline = pipeline
        except Exception as exc:
            results.append({"model": name, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "status": str(exc)})

    if best_pipeline is None:
        raise RuntimeError("Model training failed for all configured classifiers.")

    best_result = max(results, key=lambda item: item["f1"])
    best_name = best_result["model"]
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    sample_predictions = best_pipeline.predict(X_test)
    report = classification_report(y_test, sample_predictions, output_dict=True, zero_division=0)
    artifacts = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "target_column": target_column,
        "target_display": _target_display_name(target_column),
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_dtypes": {column: str(dtype) for column, dtype in X.dtypes.items()},
        "categorical_options": _categorical_options(X[categorical_features]),
        "feature_controls": _feature_controls(X.columns.tolist(), numeric_features, categorical_features),
        "class_labels": class_labels,
        "fraud_class": fraud_class,
        "best_model": best_name,
        "best_accuracy": best_result["accuracy"],
        "best_precision": best_result["precision"],
        "best_recall": best_result["recall"],
        "best_f1": best_result["f1"],
        "results": results,
        "classification_report": report,
        "feature_importance": _extract_feature_importance(best_pipeline, X.columns.tolist()),
        "numeric_stats": _numeric_stats(X[numeric_features]),
    }
    with ARTIFACT_PATH.open("wb") as handle:
        pickle.dump(artifacts, handle)
    METRICS_PATH.write_text(json.dumps(artifacts, indent=2, default=str), encoding="utf-8")
    return artifacts


def get_training_status():
    if not METRICS_PATH.exists():
        return {"trained": False, "message": "No model has been trained yet."}
    status = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    status.setdefault("target_display", _target_display_name(status.get("target_column", "target")))
    return {"trained": True, **status}


def _clean_columns(df):
    df = df.copy()
    df.columns = [str(column).strip().replace(" ", "_").lower() for column in df.columns]
    df = df.dropna(axis=1, how="all")
    return df


def _choose_target_column(df):
    for preferred in ("isfraud", "is_fraud", "fraud", "fraudulent"):
        if preferred in df.columns:
            if df[preferred].dropna().nunique() >= 2:
                return preferred
            raise ValueError(
                f"'{preferred}' was detected as the fraud target, but the selected CSV rows contain only one class. "
                "Include both fraud and legitimate transactions before training."
            )

    scored = []
    row_count = len(df)
    for column in df.columns:
        series = df[column].dropna()
        unique_count = series.nunique()
        if unique_count < 2:
            continue
        unique_ratio = unique_count / max(row_count, 1)
        hint_score = 2.0 if any(hint in column for hint in TARGET_HINTS) else 0.0
        classification_score = 1.0 if unique_count <= min(20, max(2, row_count * 0.1)) else 0.0
        balance_score = 1.0 - abs(series.value_counts(normalize=True).max() - 0.5)
        scored.append((hint_score + classification_score + balance_score - unique_ratio, column))
    if not scored:
        raise ValueError("Could not identify a classification target column.")
    return sorted(scored, reverse=True)[0][1]


def _choose_feature_columns(df, target_column):
    columns = []
    for column in df.columns:
        if column == target_column:
            continue
        lower = column.lower()
        if any(hint in lower for hint in DROP_HINTS):
            continue
        if df[column].nunique(dropna=True) <= 1:
            continue
        if df[column].nunique(dropna=True) / max(len(df), 1) > 0.9 and df[column].dtype == object:
            continue
        columns.append(column)
    return columns


def _target_display_name(target_column):
    normalized = target_column.lower().replace("_", "")
    if normalized in {"isfraud", "fraud", "fraudulent"}:
        return "Fraud Detection"
    return target_column.replace("_", " ").title()


def _encode_target(series):
    normalized = series.astype(str).str.strip().str.lower()
    fraud_like = normalized.str.contains("fraud|yes|true|chargeback|suspicious|1", regex=True, na=False)
    if fraud_like.any() and (~fraud_like).any():
        labels = pd.Series(np.where(fraud_like, 1, 0), index=series.index)
        return labels, {0: "Legitimate", 1: "Fraud"}, 1

    codes, uniques = pd.factorize(series)
    labels = pd.Series(codes, index=series.index)
    class_labels = {int(index): str(value) for index, value in enumerate(uniques)}
    fraud_class = _infer_fraud_class(class_labels)
    return labels, class_labels, fraud_class


def _infer_fraud_class(class_labels):
    for key, value in class_labels.items():
        if any(hint in value.lower() for hint in ("fraud", "suspicious", "chargeback", "risk")):
            return int(key)
    return int(max(class_labels.keys()))


def _feature_engineering(X):
    engineered = X.copy()
    numeric = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    if len(numeric) >= 2:
        first, second = numeric[:2]
        engineered[f"{first}_to_{second}_ratio"] = engineered[first] / engineered[second].replace(0, np.nan)
    for column in numeric:
        if any(token in column for token in ("amount", "price", "value", "balance")):
            engineered[f"log_{column}"] = np.log1p(pd.to_numeric(engineered[column], errors="coerce").clip(lower=0))
    return engineered


def _build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _candidate_models(train_size, class_count):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=max(3, min(11, int(np.sqrt(max(train_size, 1)))))),
    }
    try:
        from xgboost import XGBClassifier

        if class_count == 2:
            models["XGBoost"] = XGBClassifier(
                n_estimators=160,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            )
    except Exception:
        pass
    return models


def _extract_feature_importance(pipeline, original_features):
    model = pipeline.named_steps["model"]
    importance = {}
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        raw = np.mean(np.abs(model.coef_), axis=0)
    else:
        return {feature: 1 / max(len(original_features), 1) for feature in original_features}

    transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    total_by_original = {feature: 0.0 for feature in original_features}
    for name, value in zip(transformed_names, raw):
        for feature in original_features:
            if name.endswith(feature) or f"__{feature}" in name or f"__{feature}_" in name:
                total_by_original[feature] += float(abs(value))
                break
    total = sum(total_by_original.values()) or 1.0
    return {feature: value / total for feature, value in total_by_original.items()}


def _numeric_stats(frame):
    stats = {}
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        stats[column] = {
            "median": float(series.median()) if pd.notna(series.median()) else 0.0,
            "iqr": float(q3 - q1) if pd.notna(q3 - q1) and q3 != q1 else 1.0,
        }
    return stats


def _categorical_options(frame):
    options = {}
    for column in frame.columns:
        values = frame[column].dropna().astype(str).str.strip()
        top_values = values[values != ""].value_counts().head(25).index.tolist()
        options[column] = top_values
    return options


def _feature_controls(features, numeric_features, categorical_features):
    controls = {}
    for feature in features:
        lower = feature.lower()
        if any(token in lower for token in ("date", "dob")):
            controls[feature] = "date"
        elif any(token in lower for token in ("time", "timestamp")):
            controls[feature] = "datetime-local"
        elif feature in categorical_features or any(token in lower for token in ("type", "category", "status", "mode")):
            controls[feature] = "select"
        elif feature in numeric_features:
            controls[feature] = "number"
        else:
            controls[feature] = "text"
    return controls
