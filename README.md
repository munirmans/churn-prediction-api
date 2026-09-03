# Churn Prediction API

FastAPI service that predicts telecom customer churn from a trained scikit-learn pipeline (logistic regression, `class_weight="balanced"`, L1 regularization, tuned via cross-validated grid search).

**Live demo:** [churn-prediction-api.fly.dev/docs](https://churn-prediction-api.fly.dev/docs) — interactive Swagger UI, try `POST /predict` with sample customer data.

## Stack

- **Model**: scikit-learn `Pipeline` (`ColumnTransformer` + `LogisticRegression`), trained on the Telco Customer Churn dataset
- **API**: FastAPI + Pydantic for request validation
- **Containerized**: Docker
- **Deployed**: [fly.io](https://fly.io)

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs` to test the `POST /predict` endpoint interactively.

## Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
