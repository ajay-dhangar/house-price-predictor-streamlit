import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- MODEL LOADER (SAFE) ---------------- #
@st.cache_resource
def load_or_train_model():
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"

    # If model exists → load it
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    # Else → train model (Cloud-safe)
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["Price"] = housing.target

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler


model, scaler = load_or_train_model()

# ---------------- HACKER / AI CSS ---------------- #
st.markdown("""
<style>
body {
    background-color: #020617;
    color: #00f5ff;
}
.main {
    background: radial-gradient(circle at top, #020617, #000000);
}
h1, h2, h3 {
    color: #00f5ff;
    text-shadow: 0 0 10px #00f5ff;
}
.stButton > button {
    background: linear-gradient(90deg, #00f5ff, #38bdf8);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 0 20px #00f5ff;
}
.stButton > button:hover {
    transform: scale(1.05);
}
.stNumberInput input {
    background-color: #020617;
    color: #00f5ff;
}
hr {
    border: 1px solid #00f5ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("## 🧠 AI HOUSE PRICE PREDICTOR")
st.markdown("**> Autonomous ML System | Hacker Mode Enabled**")
st.markdown("---")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.markdown("## ⚙ SYSTEM INPUT PARAMETERS")

MedInc = st.sidebar.number_input("Median Income", value=5.0)
HouseAge = st.sidebar.number_input("House Age", value=25.0)
AveRooms = st.sidebar.number_input("Average Rooms", value=6.0)
AveBedrms = st.sidebar.number_input("Average Bedrooms", value=1.0)
Population = st.sidebar.number_input("Population", value=300.0)
AveOccup = st.sidebar.number_input("Average Occupancy", value=3.0)
Latitude = st.sidebar.number_input("Latitude", value=34.05)
Longitude = st.sidebar.number_input("Longitude", value=-118.25)

# ---------------- PREDICTION ---------------- #
st.markdown("### 🔮 Prediction Console")

if st.button("🚀 EXECUTE AI PREDICTION"):
    with st.spinner("Booting neural subsystems..."):
        time.sleep(1.0)

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                           Population, AveOccup, Latitude, Longitude]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0] * 100000

    st.success("🟢 Prediction Successful")

    st.markdown(f"""
## 💰 **Estimated House Price**
### `$ {prediction:,.2f}`
""")

    st.markdown("""
```log
STATUS: OK
MODEL: Linear Regression
MODE: AUTO-TRAIN SAFE
SECURITY: STABLE
```
"""
)

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption("⚡ Built by AI • Ajay Dhangar • Machine Learning • Streamlit")
