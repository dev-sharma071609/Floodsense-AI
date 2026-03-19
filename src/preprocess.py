from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns():
    numeric_features = [
        "Latitude",
        "Longitude",
        "Rainfall (mm)",
        "Temperature (°C)",
        "Humidity (%)",
        "River Discharge (m³/s)",
        "Water Level (m)",
        "Elevation (m)",
        "Population Density",
        "Infrastructure",
        "Historical Floods"
    ]

    categorical_features = [
        "Land Cover",
        "Soil Type"
    ]

    target_column = "Risk Level"

    return numeric_features, categorical_features, target_column


def create_preprocessor():
    numeric_features, categorical_features, _ = get_feature_columns()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor