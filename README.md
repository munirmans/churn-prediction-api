# Churn Prediction API

FastAPI service that predicts telecom customer churn using a trained scikit-learn pipeline (logistic regression, `class_weight="balanced"`, L1 regularization).

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test the `POST /predict` endpoint interactively.
