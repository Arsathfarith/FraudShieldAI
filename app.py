from flask import Flask, jsonify, redirect, render_template, request, url_for

from fraudshield.monitoring import api_status, log_prediction, prediction_summary, read_prediction_logs
from fraudshield.prediction import load_artifacts, predict_transaction
from fraudshield.training import MODEL_PATH, get_training_status, train_from_csv


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.get("/")
def index():
    status = get_training_status()
    return render_template("index.html", status=status)


@app.get("/train")
def train_page():
    return render_template("train.html", status=get_training_status())


@app.post("/train")
def train_upload():
    uploaded = request.files.get("dataset")
    if not uploaded or uploaded.filename == "":
        return render_template("train.html", error="Upload a CSV dataset to train the models.", status=get_training_status())

    try:
        result = train_from_csv(uploaded)
    except Exception as exc:
        return render_template("train.html", error=str(exc), status=get_training_status())

    return render_template("train.html", result=result, status=get_training_status())


@app.get("/predict")
def predict_page():
    artifacts = load_artifacts(required=False)
    return render_template("predict.html", artifacts=artifacts)


@app.post("/predict")
def predict_submit():
    artifacts = load_artifacts(required=True)
    form_values = {feature: request.form.get(feature, "") for feature in artifacts["feature_columns"]}
    result = predict_transaction(form_values, artifacts)
    log_prediction(result)
    return render_template("result.html", result=result, artifacts=artifacts)


@app.get("/dashboard")
def dashboard():
    status = get_training_status()
    logs = read_prediction_logs()
    summary = prediction_summary(logs)
    return render_template("dashboard.html", status=status, logs=logs, prediction_summary=summary)


@app.get("/explainability")
def explainability():
    artifacts = load_artifacts(required=False)
    return render_template("explainability.html", artifacts=artifacts)


@app.get("/observability")
def observability():
    status = get_training_status()
    logs = read_prediction_logs()
    return render_template("observability.html", status=status, logs=logs, api=api_status())


@app.get("/api/status")
def api_status_route():
    payload = api_status()
    payload["model_loaded"] = MODEL_PATH.exists()
    return jsonify(payload)


@app.get("/api/metrics")
def api_metrics():
    return jsonify(get_training_status())


@app.errorhandler(404)
def not_found(_):
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
