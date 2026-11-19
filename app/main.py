import mlflow
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "model"
model = mlflow.pyfunc.load_model(str(MODEL_PATH))


class PredictionInput(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(input_data: PredictionInput):
    features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": prediction.tolist()}
