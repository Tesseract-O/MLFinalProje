import joblib
import pandas as pd
import streamlit as st

artifact = joblib.load("attendance_model.joblib")
model, default_threshold = artifact['model'], artifact['threshold']

st.title('Fitness Class Attendance Predictor')

months = st.number_input('Months as member', min_value=1, max_value=60, value=1)
weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, value=80.0)
days_before = st.number_input('Days booked in advance', min_value=1, max_value=30, value=7)
day = st.selectbox('Day of week', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
time = st.selectbox('Time', ['AM', 'PM'])
category = st.selectbox('Class type', ['Strength', 'HIIT', 'Cycling', 'Yoga', 'Aqua', 'unknown'])

threshold = st.slider(
    'Decision threshold',
    0.1, 0.9, float(default_threshold),
    help='Lower = predict "attend" more often (fewer missed attendees, '
         'more false alarms). Higher = predict "attend" only when confident.'
)

if st.button('Predict'):
    # column names must match the training data exactly
    row = pd.DataFrame([{
        'months_as_member': months,
        'weight': weight,
        'days_before': days_before,
        'day_of_week': day,
        'time': time,
        'category': category,
        'is_weekend': int(day in ['Sat', 'Sun']),
        'is_wednesday': int(day == 'Wed'),
    }])

    prob = model.predict_proba(row)[0, 1]
    skip_prob = 1 - prob

    st.metric('Probability of attending', f'{prob:.0%}')
    st.progress(float(prob))

    if prob >= threshold:
        st.success(f'Likely to attend ({prob:.0%} chance)')
    else:
        st.warning(f'Likely to skip ({skip_prob:.0%} chance of no-show)')

    st.caption(
        f'The member is predicted to attend if the probability is at least '
        f'{threshold:.0%}. Change the threshold above to make the prediction '
        f'more or less cautious.'
    )

st.caption('This is a statistical estimate based on 1,500 past bookings, not a guarantee.')
