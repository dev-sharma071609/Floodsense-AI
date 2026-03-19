import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import create_preprocessor, get_feature_columns


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 50}")
    print(f"MODEL: {name}")
    print(f"{'=' * 50}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, pipeline


def main():
    data_path = os.path.join("data", "processed", "flood_risk_engineered.csv")
    model_path = os.path.join("models", "best_flood_model.pkl")

    df = pd.read_csv(data_path)

    print("Engineered dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    numeric_features, categorical_features, target_column = get_feature_columns()
    feature_columns = numeric_features + categorical_features

    print("\nMissing values:")
    print(df[feature_columns + [target_column]].isnull().sum())

    print("\nTarget distribution:")
    print(df[target_column].value_counts())

    df = df.dropna()

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    preprocessor = create_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )
    }

    best_accuracy = 0
    best_pipeline = None
    best_model_name = ""

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        accuracy, trained_pipeline = evaluate_model(
            name, pipeline, X_train, X_test, y_train, y_test
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pipeline = trained_pipeline
            best_model_name = name

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()