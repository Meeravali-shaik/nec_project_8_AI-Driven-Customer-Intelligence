import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(
"customer_dataset_100.csv"
)

X = df[
[
'Tenure_Months',
'Last_Purchase_Days',
'Customer_Satisfaction',
'Purchase_Frequency',
'Product_Expiry_Days'
]
]

y = df['Churn_Status']

model = RandomForestClassifier(
n_estimators=200,
random_state=42
)

model.fit(X,y)

joblib.dump(
model,
"models/churn_model.pkl"
)

print("Churn Model Saved")