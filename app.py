import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Multi-Domain Predictive Health & Analytics System",
    layout="wide"
)

st.title("Unified Machine Learning & Clinical Prediction Suite")

# Navigation Sidebar
option = st.sidebar.selectbox(
    "Select Prediction Model",
    [
        "Diabetes Risk Prediction",
        "Heart Disease Risk Prediction",
        "USA Housing Price Prediction",
        "Liver Disease Detection",
        "Chronic Kidney Disease Detection",
        "Parkinson's Voice Analysis"
    ]
)

# ---------------------------------------------------------
# 1. Diabetes Risk Prediction
# ---------------------------------------------------------
if option == "Diabetes Risk Prediction":
    st.header("Diabetes Risk Prediction Model")
    
    @st.cache_resource
    def load_and_train_diabetes():
        df = pd.read_csv('diabetes.csv')
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X.columns

    model, cols = load_and_train_diabetes()
    
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=33)
        
    if st.button("Predict Diabetes Risk"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        
        if pred == 1:
            st.error(f"High Risk of Diabetes detected (Confidence: {prob:.2f}%)")
        else:
            st.success(f"Low Risk of Diabetes (Confidence: {100 - prob:.2f}%)")

# ---------------------------------------------------------
# 2. Heart Disease Risk Prediction
# ---------------------------------------------------------
elif option == "Heart Disease Risk Prediction":
    st.header("Heart Disease Risk Prediction Model")
    
    @st.cache_resource
    def load_and_train_heart():
        df = pd.read_csv('heart_disease_data.csv')
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = load_and_train_heart()
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 50, 220, 120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 50, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0=normal; 1=fixed defect; 2=reversable defect)", [0, 1, 2, 3])

    if st.button("Predict Heart Disease Status"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        
        if pred == 1:
            st.error(f"Presence of Heart Disease detected (Confidence: {prob:.2f}%)")
        else:
            st.success(f"No Heart Disease detected (Confidence: {100 - prob:.2f}%)")

# ---------------------------------------------------------
# 3. USA Housing Price Prediction
# ---------------------------------------------------------
elif option == "USA Housing Price Prediction":
    st.header("USA Housing Price Regression Model")
    
    @st.cache_resource
    def load_and_train_housing():
        df = pd.read_csv('USA_Housing.csv')
        X = df.drop(columns=['Price', 'Address'])
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = load_and_train_housing()
    
    income = st.number_input("Avg. Area Income ($)", 10000.0, 150000.0, 68000.0)
    house_age = st.number_input("Avg. Area House Age (Years)", 1.0, 20.0, 6.0)
    rooms = st.number_input("Avg. Area Number of Rooms", 1.0, 15.0, 7.0)
    bedrooms = st.number_input("Avg. Area Number of Bedrooms", 1.0, 10.0, 4.0)
    pop = st.number_input("Area Population", 1000.0, 100000.0, 35000.0)

    if st.button("Predict Valuation"):
        input_data = np.array([[income, house_age, rooms, bedrooms, pop]])
        pred = model.predict(input_data)[0]
        st.success(f"Estimated Market Valuation: **${pred:,.2f}**")

# ---------------------------------------------------------
# 4. Liver Disease Detection
# ---------------------------------------------------------
elif option == "Liver Disease Detection":
    st.header("Liver Disease Classifier")
    
    @st.cache_resource
    def load_and_train_liver():
        df = pd.read_csv('indian_liver_patient.csv')
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
        df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())
        X = df.drop(columns=['Dataset'])
        y = df['Dataset']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = load_and_train_liver()
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_num = 1 if gender == "Male" else 0
        total_b = st.number_input("Total Bilirubin", 0.0, 80.0, 1.0)
        direct_b = st.number_input("Direct Bilirubin", 0.0, 30.0, 0.3)
        alk_p = st.number_input("Alkaline Phosphatase", 10, 3000, 200)
    with col2:
        ala_a = st.number_input("Alamine Aminotransferase", 10, 2000, 30)
        asp_a = st.number_input("Aspartate Aminotransferase", 10, 5000, 35)
        total_p = st.number_input("Total Proteins", 1.0, 10.0, 6.5)
        alb = st.number_input("Albumin", 0.5, 6.0, 3.0)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.1, 3.0, 0.9)

    if st.button("Evaluate Liver Health"):
        input_data = np.array([[age, gender_num, total_b, direct_b, alk_p, ala_a, asp_a, total_p, alb, ag_ratio]])
        pred = model.predict(input_data)[0]
        
        if pred == 1:
            st.error("Classified as Liver Patient (Positive Diagnosis)")
        else:
            st.success("Classified as Non-Liver Patient (Negative Diagnosis)")

# ---------------------------------------------------------
# 5. Chronic Kidney Disease Detection
# ---------------------------------------------------------
elif option == "Chronic Kidney Disease Detection":
    st.header("Chronic Kidney Disease Classifier")
    
    @st.cache_resource
    def load_and_train_kidney():
        df = pd.read_csv('kidney_disease.csv')
        df = df.drop(columns=['id'])
        # Clean target column values
        df['classification'] = df['classification'].astype(str).str.strip().replace({'ckd\t': 'ckd'})
        
        # Numeric conversions and imputation
        for col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            
        # Categorical encoding
        cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for col in cat_cols:
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
            
        X = df.drop(columns=['classification'])
        y = df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, X.columns

    model, feature_names = load_and_train_kidney()
    
    st.write("Enter Biomarker Readings:")
    age = st.number_input("Age", 1, 100, 50)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    al = st.selectbox("Albumin Level (0-5)", [0, 1, 2, 3, 4, 5])
    su = st.selectbox("Sugar Level (0-5)", [0, 1, 2, 3, 4, 5])
    sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 1.2)
    hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, 12.5)

    if st.button("Diagnose Kidney Status"):
        # Fill mean defaults for non-interactive features
        sample = np.zeros(len(feature_names))
        sample[0] = age
        sample[1] = bp
        sample[2] = sg
        sample[3] = al
        sample[4] = su
        sample[10] = sc
        sample[13] = hemo
        
        pred = model.predict([sample])[0]
        if pred == 1:
            st.error("Prediction: High Probability of Chronic Kidney Disease (CKD)")
        else:
            st.success("Prediction: Low Probability of Chronic Kidney Disease (Not CKD)")

# ---------------------------------------------------------
# 6. Parkinson's Voice Analysis
# ---------------------------------------------------------
elif option == "Parkinson's Voice Analysis":
    st.header("Parkinson's Voice Signal Classifier")
    
    @st.cache_resource
    def load_and_train_parkinsons():
        df = pd.read_csv('parkinsons.data')
        X = df.drop(columns=['name', 'status'])
        y = df['status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X.columns

    model, feature_names = load_and_train_parkinsons()
    
    col1, col2 = st.columns(2)
    with col1:
        fo = st.number_input("MDVP:Fo (Hz) - Average Pitch", 80.0, 300.0, 150.0)
        fhi = st.number_input("MDVP:Fhi (Hz) - Maximum Pitch", 100.0, 600.0, 200.0)
        flo = st.number_input("MDVP:Flo (Hz) - Minimum Pitch", 50.0, 300.0, 100.0)
        jitter = st.number_input("MDVP:Jitter (%)", 0.001, 0.05, 0.006, format="%.5f")
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", 0.005, 0.2, 0.03, format="%.5f")
        hnr = st.number_input("HNR (Harmonics-to-Noise Ratio)", 0.0, 40.0, 22.0)
        rpde = st.number_input("RPDE Value", 0.0, 1.0, 0.5)
        dfa = st.number_input("DFA Value", 0.0, 1.0, 0.6)

    if st.button("Analyze Voice Frequency"):
        sample = np.zeros(len(feature_names))
        sample[0] = fo
        sample[1] = fhi
        sample[2] = flo
        sample[3] = jitter
        sample[8] = shimmer
        sample[15] = hnr
        sample[17] = rpde
        sample[18] = dfa
        
        pred = model.predict([sample])[0]
        prob = model.predict_proba([sample])[0][1] * 100
        
        if pred == 1:
            st.error(f"Voice Metrics Indicate Positive Signs of Parkinson's (Confidence: {prob:.2f}%)")
        else:
            st.success(f"Voice Metrics Within Healthy Control Parameters (Confidence: {100 - prob:.2f}%)")
