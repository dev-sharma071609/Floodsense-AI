import os
import pandas as pd

def main():
    data_path = os.path.join("data", "raw", "flood_risk_dataset_india.csv")
    df = pd.read_csv(data_path)

    print("Dataset Shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nTarget distribution:")
    print(df["Flood Occurred"].value_counts(normalize=True))

    print("\nNumeric summary:")
    print(df.describe())

    print("\nCategorical unique values:")
    for col in ["Land Cover", "Soil Type"]:
        print(f"\n{col}:")
        print(df[col].value_counts())

    print("\nCorrelation with Flood Occurred:")
    numeric_df = df.select_dtypes(include=["number"])
    print(numeric_df.corr(numeric_only=True)["Flood Occurred"].sort_values(ascending=False))

if __name__ == "__main__":
    main()