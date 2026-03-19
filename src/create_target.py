import os
import pandas as pd


def minmax(series):
    return (series - series.min()) / (series.max() - series.min())


def create_risk_score(df):
    rainfall_norm = minmax(df["Rainfall (mm)"])
    water_level_norm = minmax(df["Water Level (m)"])
    discharge_norm = minmax(df["River Discharge (m³/s)"])
    humidity_norm = minmax(df["Humidity (%)"])
    historical_norm = minmax(df["Historical Floods"])
    infrastructure_norm = minmax(df["Infrastructure"])

    elevation_norm = minmax(df["Elevation (m)"])
    elevation_risk = 1 - elevation_norm

    score = (
        rainfall_norm * 25 +
        water_level_norm * 25 +
        discharge_norm * 20 +
        elevation_risk * 10 +
        humidity_norm * 10 +
        historical_norm * 10
    )

    land_cover_map = {
        "Urban": 8,
        "Water Body": 10,
        "Agricultural": 5,
        "Forest": 2,
        "Desert": 1
    }
    land_cover_bonus = df["Land Cover"].map(land_cover_map).fillna(0)

    soil_type_map = {
        "Clay": 8,
        "Peat": 7,
        "Silt": 6,
        "Loam": 4,
        "Sandy": 2
    }
    soil_bonus = df["Soil Type"].map(soil_type_map).fillna(0)

    infrastructure_reduction = infrastructure_norm * 8

    final_score = score + land_cover_bonus + soil_bonus - infrastructure_reduction
    final_score = final_score.clip(0, 100)

    return final_score


def score_to_level(score):
    if score < 30:
        return "Low"
    elif score < 50:
        return "Moderate"
    elif score < 70:
        return "High"
    else:
        return "Severe"


def main():
    input_path = os.path.join("data", "raw", "flood_risk_dataset_india.csv")
    output_path = os.path.join("data", "processed", "flood_risk_engineered.csv")

    df = pd.read_csv(input_path)

    df["Risk Score"] = create_risk_score(df)
    df["Risk Level"] = df["Risk Score"].apply(score_to_level)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Engineered dataset created successfully.")
    print(f"Saved to: {output_path}")

    print("\nRisk Level distribution:")
    print(df["Risk Level"].value_counts())

    print("\nSample rows:")
    print(
        df[
            [
                "Rainfall (mm)",
                "Water Level (m)",
                "River Discharge (m³/s)",
                "Elevation (m)",
                "Humidity (%)",
                "Historical Floods",
                "Land Cover",
                "Soil Type",
                "Infrastructure",
                "Risk Score",
                "Risk Level",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()