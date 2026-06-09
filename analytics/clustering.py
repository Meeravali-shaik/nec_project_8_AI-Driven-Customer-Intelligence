import joblib
import pandas as pd

model = joblib.load(
"models/cluster_model.pkl"
)

scaler = joblib.load(
"models/cluster_scaler.pkl"
)

def get_segments(df):

    X = scaler.transform(
        df[
        [
        'Age',
        'Annual_Income',
        'Purchase_Amount',
        'Purchase_Frequency'
        ]
        ]
    )

    df['Cluster'] = model.predict(X)

    return df