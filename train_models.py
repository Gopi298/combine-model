import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_diabetes_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    with open("diabetes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("diabetes_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Diabetes model trained and saved.")


def train_heart_model():
    df = pd.read_csv("heart_disease_data.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    model.fit(X_train_scaled, y_train)

    with open("heart_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("heart_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Heart Failure model trained and saved.")


if __name__ == "__main__":
    train_diabetes_model()
    train_heart_model()
