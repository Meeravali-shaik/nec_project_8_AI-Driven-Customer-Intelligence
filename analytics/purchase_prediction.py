import joblib

model = joblib.load(
"models/purchase_model.pkl"
)

def predict_product(data):

    return model.predict(
    [data]
    )[0]