import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
