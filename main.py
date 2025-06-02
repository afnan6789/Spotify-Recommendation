from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model from the pickle file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# 👋 Root endpoint to avoid 404 at "/"
@app.get("/")
def root():
    return {"message": "Welcome to the Music Like Prediction API! Go to /docs for Swagger UI."}

# Define input data model using Pydantic
class MusicFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int

# Define the prediction endpoint
@app.post("/predict")
def predict(features: MusicFeatures):
    # Convert input to 2D array for model
    input_data = np.array([[ 
        features.danceability,
        features.energy,
        features.key,
        features.loudness,
        features.mode,
        features.speechiness,
        features.acousticness,
        features.instrumentalness,
        features.liveness,
        features.valence,
        features.tempo,
        features.duration_ms,
        features.time_signature
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return {"prediction": int(prediction)}  # 0 or 1
