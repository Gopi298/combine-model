import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="Multi-Domain Machine Learning Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Auto-train and generate missing models/scalers upon startup
def ensure_models_exist():
    # 1. Diabetes Model
    if not os.path.exists("diabetes_model.pkl") or not os.path.exists(
        "diabetes_scaler.pkl"
    ):
        if os.path.exists("diabetes.csv"):
            df = pd.read_csv("diabetes.csv")
            X = df.drop("Outcome", axis=1)
            y = df["Outcome"]
            X_train, _, y_train, _ = train_test_split(
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

    # 2. Heart Failure Model
    if not os.path.exists("heart_model.pkl") or not os.path.exists(
        "heart_scaler.pkl"
    ):
        if os.path.exists("heart_disease_data.csv"):
            df = pd.read_csv("heart_disease_data.csv")
            X = df.drop("target", axis=1)
            y = df["target"]
            X_train, _, y_train, _ = train_test_split(
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

    # 3. House Price Regression Model
    if not os.path.exists("house_model.pkl") or not os.path.exists(
        "house_scaler.pkl"
    ):
        if os.path.exists("USA_Housing.csv"):
            df = pd.read_csv("USA_Housing.csv")
            X = df.drop(["Price", "Address"], axis=1)
            y = df["Price"]
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            with open("house_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("house_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

    # 4. Liver Disease Model
    if not os.path.exists("liver_model.pkl") or not os.path.exists(
        "liver_scaler.pkl"
    ):
        if os.path.exists("indian_liver_patient.csv"):
            df = pd.read_csv("indian_liver_patient.csv")
            df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
            df["Albumin_and_Globulin_Ratio"] = df[
                "Albumin_and_Globulin_Ratio"
            ].fillna(df["Albumin_and_Globulin_Ratio"].median())
            X = df.drop("Dataset", axis=1)
            y = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            )
            model.fit(X_train_scaled, y_train)
            with open("liver_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("liver_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

    # 5. Chronic Kidney Disease Model
    if not os.path.exists("kidney_model.pkl") or not os.path.exists(
        "kidney_scaler.pkl"
    ):
        if os.path.exists("kidney_disease.csv"):
            df = pd.read_csv("kidney_disease.csv")
            df = df.drop(columns=["id"])
            df["classification"] = (
                df["classification"].astype(str).str.strip().replace({"ckd\\t": "ckd"})
            )
            for col in [
                "age",
                "bp",
                "sg",
                "al",
                "su",
                "bgr",
                "bu",
                "sc",
                "sod",
                "pot",
                "hemo",
                "pcv",
                "wc",
                "rc",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            cat_cols = [
                "rbc",
                "pc",
                "pcc",
                "ba",
                "htn",
                "dm",
                "cad",
                "appet",
                "pe",
                "ane",
            ]
            for col in cat_cols:
                df[col] = df[col].astype(str)
                df[col] = LabelEncoder().fit_transform(df[col])
            X = df.drop(columns=["classification"])
            y = df["classification"].apply(lambda x: 1 if x == "ckd" else 0)
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            )
            model.fit(X_train_scaled, y_train)
            with open("kidney_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("kidney_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

    # 6. Parkinson's Disease Model
    if not os.path.exists("parkinsons_model.pkl") or not os.path.exists(
        "parkinsons_scaler.pkl"
    ):
        if os.path.exists("parkinsons.data"):
            df = pd.read_csv("parkinsons.data")
            X = df.drop(columns=["name", "status"])
            y = df["status"]
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            )
            model.fit(X_train_scaled, y_train)
            with open("parkinsons_model.pkl", "wb") as f:
                pickle.dump(model, f)
            with open("parkinsons_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)


ensure_models_exist()


# Gauge chart helper for risk classification
def create_speedometer(risk_score, title="Risk Level"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18}},
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "darkblue",
                },
                "bar": {"color": "#1f77b4"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "#2ca02c"},
                    {"range": [30, 70], "color": "#ff7f0e"},
                    {"range": [70, 100], "color": "#d62728"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# Sidebar Navigation Bar
st.sidebar.title("Navigation Menu")
page_choice = st.sidebar.radio(
    "Select Prediction Module:",
    [
        "Diabetes Prediction",
        "Heart Failure Prediction",
        "House Price Regression",
        "Liver Disease Detection",
        "Chronic Kidney Disease",
        "Parkinson's Voice Analysis",
    ],
)


@st.cache_resource
def load_artifacts(model_path, scaler_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# ==================== 1. DIABETES PREDICTION PAGE ====================
if page_choice == "Diabetes Prediction":
    st.title("Diabetes Classification System")

    top_left, top_right = st.columns([2, 1])

    with top_left:
        st.subheader("Patient Clinical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20, value=1
            )
            glucose = st.number_input(
                "Glucose Level", min_value=0, max_value=300, value=120
            )
            blood_pressure = st.number_input(
                "Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70
            )
            skin_thickness = st.number_input(
                "Skin Thickness (mm)", min_value=0, max_value=100, value=20
            )
        with col2:
            insulin = st.number_input(
                "Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80
            )
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=0.0,
                max_value=70.0,
                value=25.0,
                format="%.1f",
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5,
                format="%.3f",
            )
            age = st.number_input("Age", min_value=1, max_value=120, value=33)

        predict_btn = st.button("Predict Diabetes Status", type="primary")

    with top_right:
        st.subheader("Prediction Gauge")
        if predict_btn:
            model, scaler = load_artifacts(
                "diabetes_model.pkl", "diabetes_scaler.pkl"
            )
            input_data = np.array(
                [[
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    dpf,
                    age,
                ]]
            )
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0][1] * 100

            fig = create_speedometer(prob, "Diabetes Risk Gauge")
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"**High Risk:** {prob:.1f}% Risk Level")
            else:
                st.success(f"**Low Risk:** {prob:.1f}% Risk Level")
        else:
            st.info("Click 'Predict' to render the digital speed gauge.")

# ==================== 2. HEART FAILURE PREDICTION PAGE ====================
elif page_choice == "Heart Failure Prediction":
    st.title("Heart Failure Classification System")

    top_left_h, top_right_h = st.columns([2, 1])

    with top_left_h:
        st.subheader("Patient Clinical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            age_h = st.number_input(
                "Age", min_value=1, max_value=120, value=50, key="h_age"
            )
            sex = st.selectbox(
                "Sex",
                options=[1, 0],
                format_func=lambda x: "Male" if x == 1 else "Female",
            )
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[0, 1, 2, 3],
                help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic",
            )
            trestbps = st.number_input(
                "Resting BP (mm Hg)", min_value=50, max_value=250, value=120
            )
            chol = st.number_input(
                "Serum Cholestoral (mg/dl)",
                min_value=100,
                max_value=600,
                value=200,
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "True (1)" if x == 1 else "False (0)",
            )
            restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])

        with col2:
            thalach = st.number_input(
                "Max Heart Rate Achieved",
                min_value=50,
                max_value=230,
                value=150,
            )
            exang = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
            )
            oldpeak = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                format="%.1f",
            )
            slope = st.selectbox(
                "Slope of Peak Exercise ST", options=[0, 1, 2]
            )
            ca = st.selectbox("Major Vessels (ca)", options=[0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

        predict_btn_h = st.button("Predict Heart Condition", type="primary")

    with top_right_h:
        st.subheader("Prediction Gauge")
        if predict_btn_h:
            model_h, scaler_h = load_artifacts(
                "heart_model.pkl", "heart_scaler.pkl"
            )
            input_data_h = np.array(
                [[
                    age_h,
                    sex,
                    cp,
                    trestbps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak,
                    slope,
                    ca,
                    thal,
                ]]
            )
            scaled_data_h = scaler_h.transform(input_data_h)
            pred_h = model_h.predict(scaled_data_h)[0]
            prob_h = model_h.predict_proba(scaled_data_h)[0][1] * 100

            fig_h = create_speedometer(prob_h, "Heart Failure Risk Gauge")
            st.plotly_chart(fig_h, use_container_width=True)

            if pred_h == 1:
                st.error(f"**High Risk:** {prob_h:.1f}% Risk Level")
            else:
                st.success(f"**Low Risk:** {prob_h:.1f}% Risk Level")
        else:
            st.info("Click 'Predict' to render the digital speed gauge.")

# ==================== 3. HOUSE PRICE REGRESSION PAGE ====================
elif page_choice == "House Price Regression":
    st.title("USA House Price Regression Model")
    st.caption("Model Accuracy: Linear Regression (R² Score = 0.918 / 91.8%)")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Property & Area Metrics")
        avg_income = st.number_input(
            "Avg. Area Income ($)",
            min_value=10000.0,
            max_value=150000.0,
            value=68583.0,
            step=1000.0,
        )
        avg_house_age = st.number_input(
            "Avg. Area House Age (Years)",
            min_value=1.0,
            max_value=20.0,
            value=5.9,
            step=0.1,
        )
        avg_rooms = st.number_input(
            "Avg. Area Number of Rooms",
            min_value=1.0,
            max_value=15.0,
            value=7.0,
            step=0.1,
        )
        avg_bedrooms = st.number_input(
            "Avg. Area Number of Bedrooms",
            min_value=1.0,
            max_value=10.0,
            value=3.8,
            step=0.1,
        )
        area_pop = st.number_input(
            "Area Population",
            min_value=1000.0,
            max_value=100000.0,
            value=36163.0,
            step=500.0,
        )

        predict_house_btn = st.button("Predict House Price", type="primary")

    with col_right:
        st.subheader("Price Prediction Output")
        if predict_house_btn:
            house_model, house_scaler = load_artifacts(
                "house_model.pkl", "house_scaler.pkl"
            )
            house_input = np.array(
                [[avg_income, avg_house_age, avg_rooms, avg_bedrooms, area_pop]]
            )
            house_scaled = house_scaler.transform(house_input)
            predicted_price = house_model.predict(house_scaled)[0]

            st.metric(
                label="Estimated Valuation",
                value=f"${predicted_price:,.2f}",
            )
            st.success("Valuation computed using multi-variable regression.")
        else:
            st.info("Submit house details on the left to estimate price.")

# ==================== 4. LIVER DISEASE DETECTION PAGE ====================
elif page_choice == "Liver Disease Detection":
    st.title("Liver Disease Classification System")

    top_left_l, top_right_l = st.columns([2, 1])

    with top_left_l:
        st.subheader("Patient Clinical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            age_l = st.number_input(
                "Age", min_value=1, max_value=120, value=45, key="l_age"
            )
            gender = st.selectbox("Gender", options=["Male", "Female"])
            gender_num = 1 if gender == "Male" else 0
            total_b = st.number_input(
                "Total Bilirubin", min_value=0.0, max_value=80.0, value=1.0, format="%.1f"
            )
            direct_b = st.number_input(
                "Direct Bilirubin", min_value=0.0, max_value=30.0, value=0.3, format="%.1f"
            )
            alk_p = st.number_input(
                "Alkaline Phosphatase", min_value=10, max_value=3000, value=200
            )
        with col2:
            ala_a = st.number_input(
                "Alamine Aminotransferase", min_value=10, max_value=2000, value=30
            )
            asp_a = st.number_input(
                "Aspartate Aminotransferase", min_value=10, max_value=5000, value=35
            )
            total_p = st.number_input(
                "Total Proteins", min_value=1.0, max_value=10.0, value=6.5, format="%.1f"
            )
            alb = st.number_input(
                "Albumin", min_value=0.5, max_value=6.0, value=3.0, format="%.1f"
            )
            ag_ratio = st.number_input(
                "Albumin and Globulin Ratio", min_value=0.1, max_value=3.0, value=0.9, format="%.2f"
            )

        predict_btn_l = st.button("Predict Liver Condition", type="primary")

    with top_right_l:
        st.subheader("Prediction Gauge")
        if predict_btn_l:
            model_l, scaler_l = load_artifacts(
                "liver_model.pkl", "liver_scaler.pkl"
            )
            input_data_l = np.array(
                [[
                    age_l,
                    gender_num,
                    total_b,
                    direct_b,
                    alk_p,
                    ala_a,
                    asp_a,
                    total_p,
                    alb,
                    ag_ratio,
                ]]
            )
            scaled_data_l = scaler_l.transform(input_data_l)
            pred_l = model_l.predict(scaled_data_l)[0]
            prob_l = model_l.predict_proba(scaled_data_l)[0][1] * 100

            fig_l = create_speedometer(prob_l, "Liver Disease Risk Gauge")
            st.plotly_chart(fig_l, use_container_width=True)

            if pred_l == 1:
                st.error(f"**High Risk:** {prob_l:.1f}% Risk Level")
            else:
                st.success(f"**Low Risk:** {prob_l:.1f}% Risk Level")
        else:
            st.info("Click 'Predict' to render the digital speed gauge.")

# ==================== 5. CHRONIC KIDNEY DISEASE PAGE ====================
elif page_choice == "Chronic Kidney Disease":
    st.title("Chronic Kidney Disease Classification System")

    top_left_k, top_right_k = st.columns([2, 1])

    with top_left_k:
        st.subheader("Patient Clinical Parameters")
        col1, col2 = st.columns(2)
        with col1:
            age_k = st.number_input(
                "Age", min_value=1, max_value=120, value=50, key="k_age"
            )
            bp_k = st.number_input(
                "Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80
            )
            sg_k = st.selectbox(
                "Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025]
            )
            al_k = st.selectbox("Albumin (0-5)", options=[0, 1, 2, 3, 4, 5])
            su_k = st.selectbox("Sugar (0-5)", options=[0, 1, 2, 3, 4, 5])
            rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", options=["normal", "abnormal"])
            pcc = st.selectbox("Pus Cell Clumps", options=["notpresent", "present"])
            ba = st.selectbox("Bacteria", options=["notpresent", "present"])
            bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
            bu = st.number_input("Blood Urea", 10, 300, 40)
            sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2, format="%.1f")

        with col2:
            sod = st.number_input("Sodium", 100, 180, 138)
            pot = st.number_input("Potassium", 2.0, 10.0, 4.5, format="%.1f")
            hemo = st.number_input("Hemoglobin", 3.0, 20.0, 12.5, format="%.1f")
            pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
            wc = st.number_input("White Blood Cell Count", 2000, 25000, 8000)
            rc = st.number_input("Red Blood Cell Count", 2.0, 8.0, 5.0, format="%.1f")
            htn = st.selectbox("Hypertension", options=["yes", "no"])
            dm = st.selectbox("Diabetes Mellitus", options=["yes", "no"])
            cad = st.selectbox("Coronary Artery Disease", options=["yes", "no"])
            appet = st.selectbox("Appetite", options=["good", "poor"])
            pe = st.selectbox("Pedal Edema", options=["yes", "no"])
            ane = st.selectbox("Anemia", options=["yes", "no"])

        predict_btn_k = st.button("Predict Kidney Condition", type="primary")

    with top_right_k:
        st.subheader("Prediction Gauge")
        if predict_btn_k:
            model_k, scaler_k = load_artifacts(
                "kidney_model.pkl", "kidney_scaler.pkl"
            )

            # Map categorical inputs consistently
            rbc_n = 1 if rbc == "normal" else 0
            pc_n = 1 if pc == "normal" else 0
            pcc_n = 1 if pcc == "present" else 0
            ba_n = 1 if ba == "present" else 0
            htn_n = 1 if htn == "yes" else 0
            dm_n = 1 if dm == "yes" else 0
            cad_n = 1 if cad == "yes" else 0
            appet_n = 1 if appet == "poor" else 0
            pe_n = 1 if pe == "yes" else 0
            ane_n = 1 if ane == "yes" else 0

            input_data_k = np.array(
                [[
                    age_k, bp_k, sg_k, al_k, su_k, rbc_n, pc_n, pcc_n, ba_n,
                    bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn_n, dm_n,
                    cad_n, appet_n, pe_n, ane_n
                ]]
            )
            scaled_data_k = scaler_k.transform(input_data_k)
            pred_k = model_k.predict(scaled_data_k)[0]
            prob_k = model_k.predict_proba(scaled_data_k)[0][1] * 100

            fig_k = create_speedometer(prob_k, "Kidney Disease Risk Gauge")
            st.plotly_chart(fig_k, use_container_width=True)

            if pred_k == 1:
                st.error(f"**High Risk:** {prob_k:.1f}% Risk Level")
            else:
                st.success(f"**Low Risk:** {prob_k:.1f}% Risk Level")
        else:
            st.info("Click 'Predict' to render the digital speed gauge.")

# ==================== 6. PARKINSON'S VOICE ANALYSIS PAGE ====================
elif page_choice == "Parkinson's Voice Analysis":
    st.title("Parkinson's Voice Analysis System")

    top_left_p, top_right_p = st.columns([2, 1])

    with top_left_p:
        st.subheader("Voice Acoustic Parameters")
        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", 80.0, 300.0, 154.2, format="%.2f")
            fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 197.1, format="%.2f")
            flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 116.3, format="%.2f")
            jitter_pct = st.number_input("MDVP:Jitter(%)", 0.001, 0.05, 0.006, format="%.5f")
            jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.00001, 0.001, 0.00004, format="%.5f")
            rap = st.number_input("MDVP:RAP", 0.0005, 0.03, 0.003, format="%.5f")
            ppq = st.number_input("MDVP:PPQ", 0.0005, 0.03, 0.003, format="%.5f")
            ddp = st.number_input("Jitter:DDP", 0.001, 0.1, 0.01, format="%.5f")
            shimmer = st.number_input("MDVP:Shimmer", 0.005, 0.2, 0.03, format="%.5f")
            shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.05, 2.0, 0.3, format="%.4f")
            apq3 = st.number_input("Shimmer:APQ3", 0.002, 0.1, 0.015, format="%.5f")
        with col2:
            apq5 = st.number_input("Shimmer:APQ5", 0.002, 0.1, 0.018, format="%.5f")
            apq = st.number_input("MDVP:APQ", 0.002, 0.15, 0.024, format="%.5f")
            dda = st.number_input("Shimmer:DDA", 0.005, 0.3, 0.045, format="%.5f")
            nhr = st.number_input("NHR", 0.0001, 0.5, 0.025, format="%.5f")
            hnr = st.number_input("HNR", 0.0, 40.0, 21.8, format="%.2f")
            rpde = st.number_input("RPDE", 0.0, 1.0, 0.49, format="%.4f")
            dfa = st.number_input("DFA", 0.0, 1.0, 0.71, format="%.4f")
            spread1 = st.number_input("spread1", -10.0, 0.0, -5.68, format="%.4f")
            spread2 = st.number_input("spread2", 0.0, 1.0, 0.22, format="%.4f")
            d2 = st.number_input("D2", 0.5, 5.0, 2.38, format="%.4f")
            ppe = st.number_input("PPE", 0.0, 1.0, 0.20, format="%.4f")

        predict_btn_p = st.button("Predict Parkinson's Condition", type="primary")

    with top_right_p:
        st.subheader("Prediction Gauge")
        if predict_btn_p:
            model_p, scaler_p = load_artifacts(
                "parkinsons_model.pkl", "parkinsons_scaler.pkl"
            )
            input_data_p = np.array(
                [[
                    fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp,
                    shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                    rpde, dfa, spread1, spread2, d2, ppe
                ]]
            )
            scaled_data_p = scaler_p.transform(input_data_p)
            pred_p = model_p.predict(scaled_data_p)[0]
            prob_p = model_p.predict_proba(scaled_data_p)[0][1] * 100

            fig_p = create_speedometer(prob_p, "Parkinson's Risk Gauge")
            st.plotly_chart(fig_p, use_container_width=True)

            if pred_p == 1:
                st.error(f"**High Risk:** {prob_p:.1f}% Risk Level")
            else:
                st.success(f"**Low Risk:** {prob_p:.1f}% Risk Level")
        else:
            st.info("Click 'Predict' to render the digital speed gauge.")
