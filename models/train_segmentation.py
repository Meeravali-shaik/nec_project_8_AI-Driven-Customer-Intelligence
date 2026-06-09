import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("customer_dataset_100.csv")

features = df[
[
'Age',
'Annual_Income',
'Purchase_Amount',
'Purchase_Frequency'
]
]

scaler = StandardScaler()

X = scaler.fit_transform(features)

model = KMeans(
n_clusters=4,
random_state=42
)

model.fit(X)

joblib.dump(model,"models/cluster_model.pkl")
joblib.dump(scaler,"models/cluster_scaler.pkl")

print("Segmentation Model Saved")