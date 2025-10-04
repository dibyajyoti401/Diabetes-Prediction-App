import numpy as np
import pickle
import streamlit as st
model=pickle.load(open("model.pkl",'rb'))
scaler=pickle.load(open("scaler.pkl","rb"))

st.title("Diabetes Prediction App")

st.markdown=("fill in the values below to predict your risk of diabetes")

pregnancies=st.number_input("Pregnancies",min_value=0,max_value=20)
glucose=st.number_input("Glucose",min_value=0,max_value=200)
bp=st.number_input("BloodPressure",min_value=0,max_value=200)
skin=st.number_input("SkinThickness",min_value=0,max_value=100)
insulin=st.number_input("Insulin",min_value=0,max_value=900)
bmi=st.number_input("BMI",min_value=0.0,max_value=70.0)
dpf=st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0)
age=st.number_input("Age",min_value=1,max_value=150)

if st.button("Predict Diabetes Risk"):
    input_data=np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaler_input=scaler.transform(input_data)
    # predict probality
    prob=model.predict_proba(scaler_input)[0][1]
    risk_percentage=round(prob*100,2)
    st.write(f"Percentage Risk:  {risk_percentage}%  ")
    
    if risk_percentage < 30:
        st.success("🟢 Low Risk of Diabetes")
    elif 30 <= risk_percentage < 70:
        st.warning("🟡 Modarate Risk of Diabetes")
    else:
        st.error("🔴 High Risk of Diabetes")


if st.button("Reset"):
    st.rerun()