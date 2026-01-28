import pandas as pd
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print("[+] Loading Dataset...")

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

r2 = r2_score(y_test, model.predict(X_test_scaled))

print(f"[✔] Model Trained | R² Score: {r2:.4f}")

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("[✔] Model & Scaler Saved Successfully")
