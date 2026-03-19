import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="FloodSense AI",
    page_icon="🌊",
    layout="wide"
)


@st.cache_resource
def load_model():
    model_path = os.path.join("models", "best_flood_model.pkl")
    return joblib.load(model_path)


model = load_model()


def get_risk_message(prediction):
    messages = {
        "Low": "Current conditions suggest relatively low flood risk. Normal monitoring is sufficient.",
        "Moderate": "Conditions indicate moderate flood risk. Keep monitoring local conditions and alerts.",
        "High": "Conditions indicate high flood risk. Precautionary planning is strongly recommended.",
        "Severe": "Conditions indicate severe flood risk. Immediate caution and preparedness are strongly advised."
    }
    return messages.get(prediction, "Risk estimate generated.")


def get_factor_summary(rainfall, water_level, river_discharge, humidity, elevation, historical_floods, land_cover, soil_type):
    reasons = []

    if rainfall >= 250:
        reasons.append("Very high rainfall detected")
    elif rainfall >= 180:
        reasons.append("Elevated rainfall levels")

    if water_level >= 10:
        reasons.append("Water level is critically high")
    elif water_level >= 7:
        reasons.append("Water level is elevated")

    if river_discharge >= 7000:
        reasons.append("River discharge is very high")
    elif river_discharge >= 4500:
        reasons.append("River discharge is elevated")

    if humidity >= 85:
        reasons.append("High humidity supports flood-prone weather conditions")

    if elevation <= 150:
        reasons.append("Low elevation increases flood vulnerability")

    if historical_floods >= 4:
        reasons.append("Area has a strong history of flooding")

    if land_cover in ["Urban", "Water Body"]:
        reasons.append(f"Land cover type ({land_cover}) can increase flood sensitivity")

    if soil_type in ["Clay", "Peat", "Silt"]:
        reasons.append(f"Soil type ({soil_type}) may reduce drainage efficiency")

    if not reasons:
        reasons.append("Overall environmental conditions were used in the risk estimate")

    return reasons


def generate_timeline(top_prob, prediction):
    base = float(top_prob)

    if prediction == "Low":
        values = np.array([
            max(0.05, base - 0.12),
            max(0.05, base - 0.08),
            max(0.05, base - 0.04),
            base,
            min(1.0, base + 0.02),
            min(1.0, base + 0.04),
            min(1.0, base + 0.06)
        ])
    elif prediction == "Moderate":
        values = np.array([
            max(0.10, base - 0.10),
            max(0.10, base - 0.06),
            max(0.10, base - 0.03),
            base,
            min(1.0, base + 0.03),
            min(1.0, base + 0.05),
            min(1.0, base + 0.07)
        ])
    elif prediction == "High":
        values = np.array([
            max(0.15, base - 0.09),
            max(0.15, base - 0.05),
            max(0.15, base - 0.02),
            base,
            min(1.0, base + 0.03),
            min(1.0, base + 0.05),
            min(1.0, base + 0.08)
        ])
    else:
        values = np.array([
            max(0.20, base - 0.08),
            max(0.20, base - 0.04),
            max(0.20, base - 0.01),
            base,
            min(1.0, base + 0.03),
            min(1.0, base + 0.05),
            min(1.0, base + 0.07)
        ])

    return values


st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
    color: #f8fafc !important;
}

.stApp {
    background: linear-gradient(180deg, #020617 0%, #0b1a33 100%);
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 60%, #0ea5e9 100%);
    padding: 34px;
    border-radius: 24px;
    color: white;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 30px rgba(0,0,0,0.28);
}

.hero-title {
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #dbeafe, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 1rem;
    color: #f8fafc !important;
}

.hero-text {
    font-size: 1.06rem;
    line-height: 1.7;
    color: #e2e8f0 !important;
}

.section-title {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 14px;
    color: #ffffff !important;
}

.mini-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 16px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.20);
}

.mini-title {
    font-size: 0.98rem;
    color: #cbd5e1 !important;
    margin-bottom: 6px;
    font-weight: 600;
}

.mini-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #ffffff !important;
}

.info-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    color: #f8fafc !important;
    line-height: 1.8;
    font-size: 1.04rem;
}

label, p, span {
    color: #e5e7eb !important;
}

.stNumberInput label,
.stSlider label,
.stSelectbox label {
    color: #ffffff !important;
    font-size: 1.04rem !important;
    font-weight: 700 !important;
}

div[role="radiogroup"] * {
    color: #ffffff !important;
    font-weight: 700 !important;
}

input, textarea {
    background: #0f172a !important;
    color: #ffffff !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.20) !important;
}

/* dropdown menu fix */
div[data-baseweb="popover"] {
    background-color: #0f172a !important;
}

div[data-baseweb="menu"] {
    background-color: #0f172a !important;
}

div[role="listbox"] {
    background-color: #0f172a !important;
}

div[role="option"] {
    background-color: #0f172a !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
}

div[role="option"]:hover {
    background-color: #1e293b !important;
    color: #ffffff !important;
}

div[aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white !important;
    font-weight: 800;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    border: none;
    font-size: 1.05rem;
    transition: 0.2s ease-in-out;
    box-shadow: 0 8px 18px rgba(37,99,235,0.35);
}

div.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 28px rgba(37,99,235,0.45);
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 10px;
    border-radius: 14px;
    color: #ffffff !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
    color: #ffffff !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #38bdf8) !important;
    box-shadow: 0 0 12px rgba(59,130,246,0.35);
}

.risk-low {
    background: linear-gradient(135deg, #166534, #22c55e);
    color: white !important;
    padding: 18px;
    border-radius: 16px;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
}

.risk-moderate {
    background: linear-gradient(135deg, #ca8a04, #facc15);
    color: #111827 !important;
    padding: 18px;
    border-radius: 16px;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
}

.risk-high {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white !important;
    padding: 18px;
    border-radius: 16px;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
}

.risk-severe {
    background: linear-gradient(135deg, #7f1d1d, #450a0a);
    color: white !important;
    padding: 18px;
    border-radius: 16px;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
}

.small-note {
    color: #cbd5e1 !important;
    font-size: 0.96rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">🌊 FloodSense AI</div>
    <div class="hero-subtitle">AI-powered rainfall and flood risk early warning system</div>
    <div class="hero-text">
        This tool estimates flood risk using environmental, geographic, and infrastructure-related inputs.
        It is designed as an educational decision-support system for awareness and preparedness.
    </div>
</div>
""", unsafe_allow_html=True)

top_col1, top_col2, top_col3 = st.columns(3)
with top_col1:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Model Type</div>
        <div class="mini-value">Multiclass ML</div>
    </div>
    """, unsafe_allow_html=True)
with top_col2:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Risk Levels</div>
        <div class="mini-value">4 Classes</div>
    </div>
    """, unsafe_allow_html=True)
with top_col3:
    st.markdown("""
    <div class="mini-card">
        <div class="mini-title">Use Case</div>
        <div class="mini-value">Early Awareness</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown('<div class="section-title">⚙️ Scenario Mode</div>', unsafe_allow_html=True)
mode = st.radio("Choose input mode", ["Manual Input", "Extreme Scenario"], horizontal=True)

if mode == "Extreme Scenario":
    default_latitude = 22.5
    default_longitude = 78.5
    default_rainfall = 320.0
    default_temperature = 27.0
    default_humidity = 92.0
    default_discharge = 8200.0
    default_water_level = 12.0
    default_elevation = 80.0
    default_population_density = 2200.0
    default_infrastructure = 3
    default_historical_floods = 5
    default_land_cover = "Urban"
    default_soil_type = "Clay"
else:
    default_latitude = 25.0
    default_longitude = 80.0
    default_rainfall = 200.0
    default_temperature = 28.0
    default_humidity = 85.0
    default_discharge = 5000.0
    default_water_level = 8.0
    default_elevation = 120.0
    default_population_density = 1500.0
    default_infrastructure = 4
    default_historical_floods = 1
    default_land_cover = "Agricultural"
    default_soil_type = "Clay"

land_cover_options = ["Urban", "Water Body", "Agricultural", "Forest", "Desert"]
soil_type_options = ["Clay", "Peat", "Silt", "Loam", "Sandy"]

st.markdown('<div class="section-title">📊 Enter Location & Environmental Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("📍 Latitude", min_value=8.0, max_value=37.0, value=default_latitude, step=0.01)
    longitude = st.number_input("🧭 Longitude", min_value=68.0, max_value=98.0, value=default_longitude, step=0.01)
    rainfall = st.slider("🌧 Rainfall (mm)", 0.0, 500.0, float(default_rainfall))
    temperature = st.slider("🌡 Temperature (°C)", 0.0, 50.0, float(default_temperature))
    humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, float(default_humidity))
    river_discharge = st.slider("🌊 River Discharge (m³/s)", 0.0, 10000.0, float(default_discharge))
    land_cover = st.selectbox("🌿 Land Cover", land_cover_options, index=land_cover_options.index(default_land_cover))

with col2:
    water_level = st.slider("📈 Water Level (m)", 0.0, 15.0, float(default_water_level))
    elevation = st.slider("🏔 Elevation (m)", 0.0, 3000.0, float(default_elevation))
    population_density = st.slider("👥 Population Density", 0.0, 5000.0, float(default_population_density))
    infrastructure = st.slider("🏗 Infrastructure", 0, 10, int(default_infrastructure))
    historical_floods = st.slider("📚 Historical Floods", 0, 10, int(default_historical_floods))
    soil_type = st.selectbox("🪨 Soil Type", soil_type_options, index=soil_type_options.index(default_soil_type))

st.markdown("---")

st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
sum1, sum2, sum3, sum4 = st.columns(4)
with sum1:
    st.metric("🌧 Rainfall", f"{rainfall:.1f} mm")
with sum2:
    st.metric("📈 Water Level", f"{water_level:.1f} m")
with sum3:
    st.metric("💧 Humidity", f"{humidity:.1f}%")
with sum4:
    st.metric("🌊 Discharge", f"{river_discharge:.0f} m³/s")

predict_clicked = st.button("🚀 Predict Flood Risk", use_container_width=True)

if predict_clicked:
    with st.spinner("Analyzing flood risk..."):
        time.sleep(1)

        X = pd.DataFrame([{
            "Latitude": latitude,
            "Longitude": longitude,
            "Rainfall (mm)": rainfall,
            "Temperature (°C)": temperature,
            "Humidity (%)": humidity,
            "River Discharge (m³/s)": river_discharge,
            "Water Level (m)": water_level,
            "Elevation (m)": elevation,
            "Population Density": population_density,
            "Infrastructure": infrastructure,
            "Historical Floods": historical_floods,
            "Land Cover": land_cover,
            "Soil Type": soil_type
        }])

        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        labels = model.classes_

        prob_dict = dict(zip(labels, probs))
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

    st.success("Prediction complete")

    emoji_map = {
        "Low": "🟢",
        "Moderate": "🟡",
        "High": "🔴",
        "Severe": "🚨"
    }

    st.markdown('<div class="section-title">🚨 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"<div class='risk-{pred.lower()}'>{emoji_map[pred]} Predicted Risk Level: {pred}</div>",
        unsafe_allow_html=True
    )

    top_prob = sorted_probs[0][1]
    second_label = sorted_probs[1][0]
    second_prob = sorted_probs[1][1]

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("🔥 Top Prediction Confidence", f"{top_prob * 100:.2f}%")
    with metric_col2:
        st.metric("📌 Second Most Likely Risk", f"{second_label} ({second_prob * 100:.2f}%)")

    gauge_col, map_col = st.columns(2)

    with gauge_col:
        st.markdown("### Confidence Gauge")
        gauge_color = {
            "Low": "#22c55e",
            "Moderate": "#eab308",
            "High": "#ef4444",
            "Severe": "#7f1d1d"
        }[pred]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top_prob * 100,
            title={"text": "Prediction Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 50], "color": "#854d0e"},
                    {"range": [50, 70], "color": "#991b1b"},
                    {"range": [70, 100], "color": "#450a0a"},
                ]
            }
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "size": 16}
        )
        st.plotly_chart(fig, use_container_width=True)

    with map_col:
        st.markdown("### Location Map")
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, use_container_width=True)

    st.markdown("### Probability Breakdown")
    for label, prob in sorted_probs:
        st.write(f"**{label}: {prob * 100:.2f}%**")
        st.progress(float(prob))

    st.markdown("### 📈 Simulated 7-Day Risk Trend")
    timeline_vals = generate_timeline(top_prob, pred)
    timeline_df = pd.DataFrame({
        "Day": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        "Risk Confidence": timeline_vals
    }).set_index("Day")
    st.line_chart(timeline_df, use_container_width=True)

    st.markdown("### 🧠 Interpretation")
    st.info(get_risk_message(pred))

    st.markdown("### Why this prediction?")
    reasons = get_factor_summary(
        rainfall=rainfall,
        water_level=water_level,
        river_discharge=river_discharge,
        humidity=humidity,
        elevation=elevation,
        historical_floods=historical_floods,
        land_cover=land_cover,
        soil_type=soil_type
    )
    for reason in reasons:
        st.write(f"- {reason}")

st.markdown("---")

st.markdown('<div class="section-title">How the Model Works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
The original dataset label showed weak correlation with key hydrological features.
To address this, a domain-informed flood risk score was engineered using rainfall, water level,
river discharge, elevation, humidity, historical flood exposure, land cover, soil type, and infrastructure.
A multiclass machine learning model was then trained to estimate four risk levels:
<b>Low, Moderate, High, and Severe</b>.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Real-World Use</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
This system can be used as an educational awareness tool for flood-prone conditions, student research demonstrations,
and early preparedness discussions. It is designed to show how environmental and geographic variables can be combined
to estimate flood risk in a structured and explainable way.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
This tool is an educational risk estimation system. It does not replace official weather forecasts,
government alerts, or emergency response systems. Real-world flooding can also be influenced by
drainage failures, dam releases, sudden local storms, and terrain-specific conditions.
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p class='small-note'>Built as a portfolio-ready AI project focused on social impact, explainability, and practical interface design.</p>",
    unsafe_allow_html=True
)