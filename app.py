# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="🌾 Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# ==============================
# CUSTOM COLORFUL CSS (Constant Background)
# ==============================
page_style = """
<style>
/* Fixed Background Image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1563201515-adbe35c669c5?auto=format&fit=crop&w=1350&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Main content container */
.block-container {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.95));
    padding: 2.5rem;
    border-radius: 25px;
    box-shadow: 0 0 30px rgba(0,0,0,0.3);
    border: 2px solid rgba(255, 255, 255, 0.5);
}

/* Headings */
h1 {
    text-align: center;
    font-weight: 900;
    font-size: 2.3rem;
    background: linear-gradient(90deg, #00b09b, #96c93d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h3, label, p {
    color: #222222 !important;
    font-weight: 600;
}

/* Input field styling */
div[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    border: 1.5px solid #4CAF50;
    color: #006400 !important;  /* Dark Green text color */
    font-weight: bold;
}
input {
    color: #006400 !important;  /* Force input text to be green */
    font-weight: bold;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #ff7e5f, #feb47b);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 26px;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #43cea2, #185a9d);
}

/* Success message styling */
.stSuccess {
    background: linear-gradient(90deg, #00c9ff, #92fe9d);
    color: #003300 !important;
    font-weight: bold;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# ==============================
# Load and Train Model
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Normalize inputs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ==============================
# Streamlit Web UI
# ==============================
st.title("🌱 Crop Recommendation System")
st.markdown("### Enter soil & weather parameters to get the best crop recommendation:")

N = st.number_input("Nitrogen (N)", 0, 200, 50)
P = st.number_input("Phosphorus (P)", 0, 200, 50)
K = st.number_input("Potassium (K)", 0, 200, 50)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
if st.button("🌾 Recommend Crop"):

    # ✅ STEP 1: DEFINE MIN & MAX CONDITIONS
    MIN_CONDITIONS = {
        "N": 10,
        "P": 5,
        "K": 5,
        "temperature": 5,
        "humidity": 20,
        "ph": 4.5,
        "rainfall": 50
    }

    MAX_CONDITIONS = {
        "N": 150,
        "P": 150,
        "K": 205,
        "temperature": 40,
        "humidity": 95,
        "ph": 9.0,
        "rainfall": 300
    }

    # ✅ STEP 2: VALIDATE CONDITIONS
    if (
        N < MIN_CONDITIONS["N"] or N > MAX_CONDITIONS["N"] or
        P < MIN_CONDITIONS["P"] or P > MAX_CONDITIONS["P"] or
        K < MIN_CONDITIONS["K"] or K > MAX_CONDITIONS["K"] or
        temperature < MIN_CONDITIONS["temperature"] or temperature > MAX_CONDITIONS["temperature"] or
        humidity < MIN_CONDITIONS["humidity"] or humidity > MAX_CONDITIONS["humidity"] or
        ph < MIN_CONDITIONS["ph"] or ph > MAX_CONDITIONS["ph"] or
        rainfall < MIN_CONDITIONS["rainfall"] or rainfall > MAX_CONDITIONS["rainfall"]
    ):
        st.error("❌ Conditions are outside the safe range for crop growth.")

    # ✅ STEP 3: PREDICT ONLY IF CONDITIONS ARE VALID
    else:
        sample = pd.DataFrame(
            [[N, P, K, temperature, humidity, ph, rainfall]],
            columns=X.columns
        )
        sample_scaled = scaler.transform(sample)
        prediction = rf.predict(sample_scaled)[0]

        st.success(f"✅ *Recommended Crop:* 🌿 {prediction}")
