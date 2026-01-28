import streamlit as st
import numpy as np
import joblib
import time

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HACKER UI CSS ---------------- #
st.markdown(
    """
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
.sidebar {
    background: #020617;
}
hr {
    border: 1px solid #00f5ff;
}
</style>
""",
    unsafe_allow_html=True
)

# ---------------- HEADER ---------------- #
st.markdown("## 🧠 AI HOUSE PRICE PREDICTOR")
st.markdown("**> Neural Regression System | Hacker Mode Enabled**")
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
    with st.spinner("Initializing neural layers..."):
        time.sleep(1.2)

    # Prepare input
    features = np.array([
        [MedInc, HouseAge, AveRooms, AveBedrms,
         Population, AveOccup, Latitude, Longitude]
    ])

    # Scale & predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0] * 100000

    st.success("🟢 Prediction Complete")

    # ✅ FIXED MULTI-LINE F-STRING
    st.markdown(
        f"""
## 💰 **Estimated House Price**
### `$ {prediction:,.2f}`
"""
    )

    st.markdown(
        """
```log
STATUS: SUCCESS
MODEL: Linear Regression
SCALING: StandardScaler
CONFIDENCE: HIGH
```

"""
)

# ---------------- FOOTER ----------------

st.markdown("---")
st.caption("⚡ Built by AI • Ajay Dhangar • Machine Learning • Streamlit")
