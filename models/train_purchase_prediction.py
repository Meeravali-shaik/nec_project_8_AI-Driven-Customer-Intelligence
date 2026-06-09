import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
"customer_dataset_100.csv"
)

X = df[
[
'Age',
'Annual_Income',
'EMI',
'Purchase_Frequency',
'Website_Visits'
]
]

y = df[
'Next_Purchase_Category'
]

model = RandomForestClassifier(
n_estimators=300,
random_state=42
)

model.fit(X,y)

joblib.dump(
model,
"models/purchase_model.pkl"
)

print("Purchase Model Saved")