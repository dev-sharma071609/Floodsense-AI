import os
import joblib
import pandas as pd


def load_model():
    model_path = os.path.join("models", "best_flood_model.pkl")
    return joblib.load(model_path)


def predict_flood(sample_data: dict):
    model = load_model()
    input_df = pd.DataFrame([sample_data])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_

    return prediction, dict(zip(class_labels, probabilities))


if __name__ == "__main__":
    sample = {
        "Latitude": 25.0,
        "Longitude": 80.0,
        "Rainfall (mm)": 220.0,
        "Temperature (°C)": 28.0,
        "Humidity (%)": 88.0,
        "River Discharge (m³/s)": 5000.0,
        "Water Level (m)": 8.0,
        "Elevation (m)": 120.0,
        "Population Density": 1500.0,
        "Infrastructure": 4,
        "Historical Floods": 1,
        "Land Cover": "Urban",
        "Soil Type": "Clay"
    }

    pred, probs = predict_flood(sample)

    print("Predicted Risk Level:", pred)
    print("Probabilities:")
    for label, prob in probs.items():
        print(f"{label}: {prob:.4f}")