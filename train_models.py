import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_best_classifier(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }
    
    best_acc, best_pipeline, best_title = 0, None, ""
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        if acc > best_acc:
            best_acc, best_pipeline, best_title = acc, {"scaler": scaler, "model": model}, name

    print(f"[{model_name}] Best Model: {best_title} | Accuracy: {best_acc * 100:.2f}%")
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(best_pipeline, f)

def train_best_regressor(X, y, model_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    
    best_r2, best_pipeline, best_title = -float("inf"), None, ""
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        r2 = r2_score(y_test, model.predict(X_test_scaled))
        if r2 > best_r2:
            best_r2, best_pipeline, best_title = r2, {"scaler": scaler, "model": model}, name

    print(f"[{model_name}] Best Model: {best_title} | R2 Score: {best_r2:.4f}")
    with open(os.path.join(MODEL_DIR, f"{model_name}.pkl"), "wb") as f:
        pickle.dump(best_pipeline, f)

# --- INDIVIDUAL TRAINERS ---
def train_liver():
    path = os.path.join(DATASET_DIR, "indian_liver_patient.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
        train_best_classifier(df.drop(columns=['Dataset']), df['Dataset'], "liver_model")

def train_kidney():
    path = os.path.join(DATASET_DIR, "kidney_disease.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'id' in df.columns: df.drop(columns=['id'], inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df.fillna(df.median(), inplace=True)
        target = 'classification' if 'classification' in df.columns else df.columns[-1]
        train_best_classifier(df.drop(columns=[target]), df[target], "kidney_model")

def train_parkinsons_class():
    path = os.path.join(DATASET_DIR, "parkinsons.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'name' in df.columns: df.drop(columns=['name'], inplace=True)
        train_best_classifier(df.drop(columns=['status']), df['status'], "parkinsons_classification_model")

def train_heart_failure():
    path = os.path.join(DATASET_DIR, "heart_failure.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        target = 'DEATH_EVENT' if 'DEATH_EVENT' in df.columns else df.columns[-1]
        train_best_classifier(df.drop(columns=[target]), df[target], "heart_failure_model")

def train_diabetes():
    path = os.path.join(DATASET_DIR, "diabetes.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]
        train_best_classifier(df.drop(columns=[target]), df[target], "diabetes_model")

def train_house_price():
    path = os.path.join(DATASET_DIR, "house_price.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df.fillna(df.median(), inplace=True)
        target = 'Price' if 'Price' in df.columns else df.columns[-1]
        train_best_regressor(df.drop(columns=[target]), df[target], "house_price_model")

if __name__ == "__main__":
    train_liver()
    train_kidney()
    train_parkinsons_class()
    train_heart_failure()
    train_diabetes()
    train_house_price()
