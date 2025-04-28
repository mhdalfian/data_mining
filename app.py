from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Membuat instance FastAPI
app = FastAPI()

# Load model Random Forest dan scaler
try:
    with open('best_rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scalerfinal.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model atau scaler: {e}")

# Definisikan struktur input sesuai fitur yang kamu pakai
class InputData(BaseModel):
    year: int
    Life_Ladder: float
    Log_GDP_per_capita: float
    Social_support: float
    Healthy_life_expectancy_at_birth: float
    Freedom_to_make_life_choices: float
    Perceptions_of_corruption: float
    Positive_affect: float
    Negative_affect: float

# Endpoint prediksi
@app.post("/predict/")
def predict(data: InputData):
    try:
        # Masukkan 9 kolom
        input_features = np.array([[ 
            data.year,
            data.Life_Ladder,
            data.Log_GDP_per_capita,
            data.Social_support,
            data.Healthy_life_expectancy_at_birth,
            data.Freedom_to_make_life_choices,
            data.Perceptions_of_corruption,
            data.Positive_affect,
            data.Negative_affect
        ]])
        
        # Normalisasi
        scaled_features = scaler.transform(input_features)

        # Prediksi
        prediction = model.predict(scaled_features)

        return {
            "input": data.dict(),
            "prediction": prediction.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
