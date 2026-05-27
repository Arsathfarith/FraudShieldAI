# FraudShield AI

End-to-end Flask classification project for fraud detection. Upload a CSV dataset, automatically select a target column, train multiple classifiers, save the best model as a Pickle file, and review analytics, explainability, and observability views.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

For local experiments with XGBoost, install the optional local dependency set:

```powershell
pip install -r requirements-local.txt
```

Open `http://127.0.0.1:5000`.

## Dataset Flow

1. Go to `/train`.
2. Upload your CSV dataset.
3. The app selects the most suitable classification target, filters useful features, preprocesses data, trains classifiers, and saves:
   - `models/best_fraud_model.pkl`
   - `models/model_artifacts.pkl`
   - `models/training_metrics.json`

## Included Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- K-Nearest Neighbors
- XGBoost when suitable for the detected target

## Project Structure

```text
app.py
fraudshield/
  monitoring.py
  prediction.py
  training.py
templates/
static/
  css/
  js/
scripts/
data/
models/
logs/
```
