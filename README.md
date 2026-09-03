# Telco Customer Churn: Prediction & API

A telecom company loses customers every month and wants to know who is likely to leave next, and why, so retention teams can intervene before they do. This project analyzes 7,043 customers, identifies the strongest drivers of churn, trains a logistic regression model tuned specifically for catching at-risk customers, and ships it as a live prediction API.

**Live demo:** [churn-prediction-api.fly.dev/docs](https://churn-prediction-api.fly.dev/docs) -- interactive Swagger UI, try `POST /predict` with sample customer data.

## Key Findings

- **Contract type is the single strongest predictor of churn**, by both mutual information (0.098, roughly 60% higher than the next feature) and raw group rates. Month-to-month customers churn at 1.61x the average rate; two-year contract customers churn at just 0.11x the average rate.
- **Tenure is the strongest numeric predictor** (correlation -0.35 with churn) -- customers who have stayed longer are meaningfully less likely to leave.
- **A cluster of internet/security add-on features** (`OnlineSecurity`, `TechSupport`, `InternetService`, `OnlineBackup`, `DeviceProtection`) form a secondary tier of predictive signal, each in the 0.04-0.06 mutual-information range.
- **`gender` and `PhoneService` are essentially uninformative** (mutual information under 0.001) -- knowing a customer's gender tells you almost nothing about whether they'll churn.
- **`TotalCharges`' raw correlation is misleading on its own.** It correlates negatively with churn (-0.20) in isolation, but that's largely a byproduct of its relationship with `tenure` (`TotalCharges` accumulates as `tenure` grows). Once the model accounts for `tenure` directly, `TotalCharges`'s trained coefficient flips to a small *positive* push toward churn -- a reminder that a univariate correlation and a multivariate model coefficient can legitimately disagree.
- The model's trained coefficients independently confirm the same story the raw group-rate analysis found: `Contract=Two year` and `tenure` are the strongest protective factors; `Contract=Month-to-month`, `InternetService=Fiber optic`, and `PaymentMethod=Electronic check` are the strongest risk factors. Two different techniques (raw data patterns and trained model weights) agreeing is a good sign the model learned something real, not noise.

## Model Performance

| Model | Precision (churn) | Recall (churn) | F1 (churn) | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.69 | 0.60 | 0.64 | 0.82 | 0.86 |
| Logistic Regression (`class_weight="balanced"`, L1-regularized) | 0.52 | 0.83 | 0.64 | 0.75 | 0.86 |

Recall is the primary metric for this problem, not accuracy or precision, because the two mistakes a churn model can make have very different costs. A **false negative** -- a customer who leaves but was never flagged -- means a silent, total loss of that customer's revenue with no chance to intervene. A **false positive** -- a customer flagged as at-risk who was actually staying -- costs a wasted retention call or discount, but the customer keeps generating revenue regardless. Missing a churner is expensive; a false alarm is cheap. The model is tuned accordingly: the balanced/regularized version catches 83% of actual churners versus the baseline's 60%, at the deliberate cost of more false alarms and lower overall accuracy. AUC (0.86 either way) confirms the underlying ranking quality is unchanged -- what moved is *where the decision threshold sits*, a deliberate business choice, not a change in how good the model fundamentally is at distinguishing the two classes.

5-fold stratified cross-validation on the tuned model confirms this isn't a fluke of one lucky train/test split: mean accuracy 0.746, mean recall 0.804 across folds.

## Recommendations

1. **Prioritize retention outreach for month-to-month customers**, especially in combination with fiber-optic internet and electronic-check payment -- the three strongest churn-risk signals the model identified.
2. **Incentivize contract upgrades.** Two-year contract customers churn at roughly a tenth of the average rate; converting month-to-month customers to longer terms is the single highest-leverage lever available.
3. **Target early-tenure customers specifically.** Since tenure is protective and cumulative, the highest-risk window is the first several months of a relationship, before loyalty (and `TotalCharges`) has had time to build.
4. **Bundle security/support add-ons** (`OnlineSecurity`, `TechSupport`, `DeviceProtection`) into retention offers -- their presence is meaningfully associated with lower churn, though this analysis does not establish that adding them *causes* retention rather than reflecting a more engaged customer segment to begin with.
5. **Deprioritize demographic targeting.** `gender` carries essentially no predictive signal; retention budget aimed at demographic segments instead of behavioral/contract signals would be poorly spent.

## Deployment

This project ships as a working service, not just a notebook. The trained `scikit-learn` `Pipeline` (preprocessing + model, fit together to prevent train/test leakage) is served behind a FastAPI endpoint, containerized with Docker, and deployed on [fly.io](https://fly.io).

```
POST /predict
```
Accepts a customer's raw feature values as JSON, returns a churn prediction and probability. Try it directly at the [live demo](https://churn-prediction-api.fly.dev/docs).

## Technical Stack

Python, pandas, scikit-learn, FastAPI, Docker, fly.io

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

## Structure

```
churn-prediction-api/
├── main.py               # FastAPI app: request schema, /predict endpoint
├── churn_model.joblib     # trained scikit-learn Pipeline (preprocessing + logistic regression)
├── Dockerfile
├── requirements.txt
└── fly.toml               # fly.io deployment config
```

The full analysis -- data cleaning, exploratory feature-importance work (mutual information, correlation, risk ratios), preprocessing pipeline construction, and model tuning (regularization search, cross-validation) -- was done in a Jupyter notebook against the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and is not included in this repository, which contains only the deployed service.
