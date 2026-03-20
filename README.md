# 🌊 FloodSense AI: Smart Agro Advisory

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![ML](https://img.shields.io/badge/Machine%20Learning-Model-success?style=for-the-badge)]
[![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=for-the-badge)]

🚀 **Live App:**  
👉 https://floodsense-ai-axgfcwvxopy8yhdccdqhpp.streamlit.app/

---

## 🌍 Overview

**FloodSense AI** is an AI-powered decision-support system that:

- 🌧 Predicts **flood risk (Low → Severe)**  
- 🌾 Recommends **optimal crops for farmers**  
- 📊 Explains predictions using environmental factors  

Built using **Machine Learning + Streamlit**, this project focuses on **real-world impact in agriculture and climate resilience**.

---

## 💡 Problem Statement

Flooding affects millions of farmers every year.

Most tools:
- Only predict floods ❌  
- Don’t help farmers take action ❌  

👉 **FloodSense AI solves this gap** by providing:
- prediction + explanation + crop advisory

---

## 🧠 Features

### 🌊 Flood Prediction
- Multiclass ML model (Low, Moderate, High, Severe)
- Uses 13 environmental + geographical features

### 📊 Explainability
- Shows **why the prediction was made**
- Highlights key contributing factors

### 🌾 Crop Advisory System
- Recommends crops based on:
  - Soil type
  - Rainfall
  - Water level
  - Flood risk

### 📈 Visual Insights
- Confidence gauge  
- Risk probability breakdown  
- 7-day simulated trend  

### 🗺️ Location Mapping
- Displays prediction location on map  

---

## 🏗️ Tech Stack

- **Python**
- **Streamlit** (UI & deployment) :contentReference[oaicite:0]{index=0}  
- **Scikit-learn** (ML model)
- **Pandas / NumPy**
- **Plotly** (visualizations)

---

## ⚙️ How It Works

1. User inputs environmental data  
2. Model predicts flood risk  
3. System analyzes:
   - rainfall  
   - water level  
   - elevation  
   - soil type  
4. Generates:
   - prediction  
   - explanation  
   - crop recommendations  

---

## 🧪 Input Features

- Latitude & Longitude  
- Rainfall (mm)  
- Temperature (°C)  
- Humidity (%)  
- River Discharge  
- Water Level  
- Elevation  
- Population Density  
- Infrastructure Index  
- Historical Floods  
- Land Cover  
- Soil Type  

---

## 🌾 Crop Advisory Logic

The system combines:

- Flood tolerance  
- Soil compatibility  
- Water availability  

Example:

| Condition | Recommendation |
|----------|---------------|
| High flood | Rice, Jute |
| Low water | Millet, Sorghum |
| Clay soil | Rice, Sugarcane |

---


---

## 🚀 Run Locally

```bash
git clone https://github.com/your-username/floodsense-ai
cd floodsense-ai
pip install -r requirements.txt
streamlit run src/app.py
---


---

## 📈 Future Improvements

* 🌐 Integrate real-time weather APIs
* 📍 Add map-based predictions (Google Maps / Leaflet)
* 🤖 Improve model with larger real datasets
* 📱 Mobile-first UI optimization
* ⚠️ Add alert notification system

---

## 👨‍💻 Author

**Dev Sharma**
Aspiring AI Engineer 🚀
Building real-world AI + ML projects

---

## ⭐ Why This Project Stands Out

✔️ End-to-end ML pipeline
✔️ Real-world problem solving
✔️ Live deployment
✔️ Clean UI/UX
✔️ Strong social impact angle

---

## ❤️ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
