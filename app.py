import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Multi-Domain Machine Learning Hub", page_icon="🤖", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_pipeline(model_filename):
    path = os.path.join(MODEL_DIR, model_filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def predict(pipeline_obj, features):
    scaler = pipeline_obj["scaler"]
    model = pipeline_obj["model"]
    scaled_data = scaler.transform(np.array([features]))
    return model.predict(scaled_data)[0]

def main():
    st.title("🤖 Multi-Domain ML Prediction System")
    
    app_mode = st.sidebar.radio(
        "Select Prediction Module:",
        [
            "Heart Failure Prediction",
            "Diabetes Diagnostic",
            "House Price Estimation",
            "Liver Disease Classification",
            "Kidney Disease Classification",
            "Parkinson's Disease Classification"
        ]
    )

    # 1. HEART FAILURE
    if app_mode == "Heart Failure Prediction":
        st.header("❤️ Heart Failure Clinical Prediction")
        pipe = load_pipeline("heart_failure_model.pkl")
        if pipe:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 20, 100, 60)
                ejection_fraction = st.number_input("Ejection Fraction (%)", 10, 80, 38)
                serum_creatinine = st.number_input("Serum Creatinine", 0.5, 10.0, 1.1)
            with c2:
                high_blood_pressure = st.selectbox("High BP", [0, 1])
                time = st.number_input("Follow-up Period (days)", 1, 300, 130)
                serum_sodium = st.number_input("Serum Sodium", 100, 150, 136)

            if st.button("Predict Risk"):
                res = predict(pipe, [age, ejection_fraction, serum_creatinine, high_blood_pressure, time, serum_sodium])
                st.error("⚠️ High risk of heart failure event.") if res == 1 else st.success("✅ Low risk detected.")

    # 2. DIABETES
    elif app_mode == "Diabetes Diagnostic":
        st.header("🩸 Diabetes Prediction")
        pipe = load_pipeline("diabetes_model.pkl")
        if pipe:
            c1, c2 = st.columns(2)
            with c1:
                pregnancies = st.number_input("Pregnancies", 0, 20, 1)
                glucose = st.number_input("Glucose Level", 0, 300, 120)
                bp = st.number_input("Blood Pressure", 0, 150, 70)
                skin = st.number_input("Skin Thickness", 0, 100, 20)
            with c2:
                insulin = st.number_input("Insulin", 0, 900, 79)
                bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.38)
                age = st.number_input("Age", 1, 120, 33)

            if st.button("Predict Diabetes"):
                res = predict(pipe, [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
                st.error("⚠️ Positive for Diabetes.") if res == 1 else st.success("✅ Negative for Diabetes.")

    # 3. HOUSE PRICE
    elif app_mode == "House Price Estimation":
        st.header("🏠 Real Estate Price Estimation")
        pipe = load_pipeline("house_price_model.pkl")
        if pipe:
            c1, c2 = st.columns(2)
            with c1:
                overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 6)
                gr_liv_area = st.number_input("Living Area (sq ft)", 300, 10000, 1500)
                garage_cars = st.slider("Garage Capacity (Cars)", 0, 5, 2)
            with c2:
                total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 5000, 1000)
                full_bath = st.slider("Full Bathrooms", 0, 5, 2)
                year_built = st.number_input("Year Built", 1800, 2026, 2005)

            if st.button("Estimate Valuation"):
                res = predict(pipe, [overall_qual, gr_liv_area, garage_cars, total_bsmt_sf, full_bath, year_built])
                st.metric("Estimated Price", f"${res:,.2f}")

    # 4. LIVER DISEASE
    elif app_mode == "Liver Disease Classification":
        st.header("🧪 Liver Disease Prediction")
        pipe = load_pipeline("liver_model.pkl")
        if pipe:
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tot_bio = st.number_input("Total Bilirubin", 0.0, 100.0, 1.0)
            dir_bio = st.number_input("Direct Bilirubin", 0.0, 50.0, 0.4)
            alk_phos = st.number_input("Alkaline Phosphotase", 0, 3000, 200)
            sgpt = st.number_input("SGPT", 0, 3000, 30)
            sgot = st.number_input("SGOT", 0, 3000, 40)
            prot = st.number_input("Total Proteins", 0.0, 15.0, 6.5)
            alb = st.number_input("Albumin", 0.0, 10.0, 3.2)
            ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 5.0, 0.9)

            if st.button("Predict Liver Status"):
                res = predict(pipe, [age, 1 if gender == "Male" else 0, tot_bio, dir_bio, alk_phos, sgpt, sgot, prot, alb, ag_ratio])
                st.error("⚠️ Liver disease detected.") if res == 1 else st.success("✅ Normal liver profile.")

    # 5. KIDNEY DISEASE
    elif app_mode == "Kidney Disease Classification":
        st.header("🫘 Chronic Kidney Disease Prediction")
        pipe = load_pipeline("kidney_model.pkl")
        if pipe:
            age = st.number_input("Age", 1, 120, 50)
            bp = st.number_input("Blood Pressure", 40, 200, 80)
            sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.020)
            al = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
            su = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
            bgr = st.number_input("Blood Glucose Random", 50, 500, 120)

            if st.button("Predict Kidney Condition"):
                res = predict(pipe, [age, bp, sg, al, su, bgr])
                st.error("⚠️ Chronic Kidney Disease detected.") if res == 1 else st.success("✅ Normal kidney function.")

    # 6. PARKINSON'S
    elif app_mode == "Parkinson's Disease Classification":
        st.header("🧠 Parkinson's Disease Test")
        pipe = load_pipeline("parkinsons_classification_model.pkl")
        if pipe:
            fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 119.9)
            fhi = st.number_input("MDVP:Fhi(Hz)", 50.0, 600.0, 157.3)
            flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 74.9)
            jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.007)
            shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.043)

            if st.button("Run Classification"):
                res = predict(pipe, [fo, fhi, flo, jitter, shimmer])
                st.error("⚠️ Parkinson's indicators present.") if res == 1 else st.success("✅ Negative.")

if __name__ == "__main__":
    main()
