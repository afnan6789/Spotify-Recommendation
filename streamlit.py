import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

st.title("🎶 Music Like Prediction App")
st.write("Fill in the song details to predict whether the song will be liked.")

# Streamlit form
with st.form("prediction_form"):
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    key = st.number_input("Key", 0, 11, 5)
    loudness = st.number_input("Loudness (in dB)", -60.0, 0.0, -10.0)
    mode = st.selectbox("Mode", [0, 1])
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.number_input("Tempo (BPM)", 0.0, 250.0, 120.0)
    duration_ms = st.number_input("Duration (ms)", 10000, 600000, 200000)
    time_signature = st.selectbox("Time Signature", [3, 4, 5])

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "duration_ms": duration_ms,
            "time_signature": time_signature
        }

        # Send request to FastAPI
        try:
            response = requests.post(API_URL, json=input_data)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"🎧 Prediction: {'Liked' if prediction == 1 else 'Not Liked'}")
            else:
                st.error(f"❌ API Error: {response.text}")
        except Exception as e:
            st.error(f"⚠️ Failed to connect to FastAPI: {e}")
