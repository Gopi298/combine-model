import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

def train_liver_model():
    df = pd.read_csv('indian_liver_patient.csv')
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
    
    X = df.drop(columns=['Dataset'])
    y = df['Dataset']
    
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    with open('liver_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'imputer': imputer}, f)
        
    print(f"Liver Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

def train_kidney_model():
    df = pd.read_csv('kidney_disease.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    
    df['classification'] = df['classification'].astype(str).str.strip()
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    df.dropna(subset=['classification'], inplace=True)
    
    cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    num_cols = [c for c in df.columns if c not in cat_cols + ['classification']]
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['classification'])
    y = df_encoded['classification']
    
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    with open('kidney_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'imputer': imputer, 'columns': X.columns.tolist()}, f)
        
    print(f"Kidney Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

def train_parkinsons_model():
    df = pd.read_csv('parkinsons.data')
    X = df.drop(columns=['name', 'status'])
    y = df['status']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    with open('parkinsons_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'columns': X.columns.tolist()}, f)
        
    print(f"Parkinson's Model Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

if __name__ == "__main__":
    train_liver_model()
    train_kidney_model()
    train_parkinsons_model()
