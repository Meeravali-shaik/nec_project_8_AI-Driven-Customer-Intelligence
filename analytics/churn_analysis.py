import joblib

model = joblib.load(
"models/churn_model.pkl"
)

def predict_churn(data):

    prediction = model.predict(
    [data]
    )[0]

    probability = model.predict_proba(
    [data]
    )[0][1]

    return prediction, probability