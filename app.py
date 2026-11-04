import pandas as pd
import streamlit as st
import joblib

artifact = joblib.load("attendance_model.joblib")
model, default_threshold = artifact['model'], artifact['threshold']

st.title('Fitness Class Attendance Predictor')

months = st.number_input('Months as member', min_value=1, max_value=60, value=1)
weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, value=80.0)
day_before = st.number_input('Day booked in advance', min_value=1, max_value=30, value=7)
day = st.selectbox('Day of Week',['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
time = st.selectbox('Time',['AM','PM'])
category = st.selectbox('Class type',['Strength','HIIT','Cycling','Yoga','Aqua','unknown'])
threshold = st.slider('Decision Threshold',0.1,0.9,float(default_threshold))

if st.button('Predict'):
    row = pd.DataFrame([{
        'months_as_member': months,
        'weight': weight,
        'days_before': day_before,
        'day_of_week': day,
        'time': time,
        'category': category,
        'is_weekend': int(day in ['Sat', 'Sun']),
        'is_wednesday': int(day == 'Wed'),
    }])
    prob = model.predict_proba(row)[0,1]
    st.metric('Attendance Probability', f'{prob:.0%}')
    st.write('Likely to attend ' if prob >= threshold else 'Likely to skip')
