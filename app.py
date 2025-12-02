import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Match Outcome Predictor", layout="centered")

st.title("🏆 Sports Performance Match Outcome Predictor")
st.write("Enter player stats below to predict whether the next match will be a **WIN or LOSS**.")

# Input sliders
training_hours = st.slider("Training Hours", 0.0, 20.0, 8.0)
fitness_score = st.slider("Fitness Score", 0.0, 100.0, 75.0)
opponent_rank = st.slider("Opponent Rank (1 = toughest opponent)", 1, 100, 50)

# Predict button
if st.button("🔥 Predict Outcome"):
    new_data = np.array([[training_hours, fitness_score, opponent_rank]])
    new_data_scaled = scaler.transform(new_data)
    result = model.predict(new_data_scaled)
    prob = model.predict_proba(new_data_scaled)[0][1] * 100

    if result[0] == 1:
        st.success("🟢 Predicted Outcome: **WIN**")
    else:
        st.error("🔴 Predicted Outcome: **LOSS**")

    st.info(f"📊 Winning Probability: **{prob:.2f}%**")

