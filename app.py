import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "admission_xgboost_model.joblib")
metadata = json.loads((BASE_DIR / "models" / "model_metadata.json").read_text(encoding="utf-8"))
FEATURES = metadata["features"]

st.set_page_config(page_title="Student Admission Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Admission Prediction")
st.write("Estimate graduate admission probability using an XGBoost regression model.")

with st.sidebar:
    st.header("Model Performance")
    st.write(f"R²: {metadata['metrics']['R2']:.3f}")
    st.write(f"MAE: {metadata['metrics']['MAE']:.3f}")
    st.write(f"RMSE: {metadata['metrics']['RMSE']:.3f}")

st.subheader("Student Details")
c1,c2 = st.columns(2)
with c1:
    gre = st.slider("GRE Score",260,340,320)
    toefl = st.slider("TOEFL Score",0,120,100)
    university = st.slider("University Rating",1,5,3)
    sop = st.slider("SOP Rating",1.0,5.0,3.5,0.5)
with c2:
    lor = st.slider("LOR Rating",1.0,5.0,3.5,0.5)
    cgpa = st.slider("CGPA",0.0,10.0,8.0,0.01)
    research = st.selectbox("Research Experience",["No","Yes"])

input_df = pd.DataFrame([[gre,toefl,university,sop,lor,cgpa,1 if research=="Yes" else 0]], columns=FEATURES)

if st.button("Predict Admission Probability", use_container_width=True):
    value = float(model.predict(input_df)[0])
    value = max(0.0,min(1.0,value))
    pct = value*100
    st.success(f"Estimated Admission Probability: {pct:.2f}%")
    st.progress(int(round(pct)))
    if pct >= 80: st.info("High predicted admission probability.")
    elif pct >= 60: st.info("Moderate predicted admission probability.")
    else: st.warning("Lower predicted admission probability.")

st.caption("Educational prediction only; this is not an official university admission decision.")
