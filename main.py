from fastapi import FastAPI
import pickle
import sklearn
from data_model import Water
import pandas as pd
app = FastAPI(
    title= "Water Potability Prediction",
    description="Predicting water Potability")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to Water Potability Prediction FastAPI"

@app.post("/predict")
def model_predict(water:Water):
    sample = pd.DataFrame({
        'ph': water.ph,
        'Hardness': water.Hardness,
        'Solids': water.Solids,
        'Chloramines': water.Chloramines,
        'Sulfate': water.Sulfate,
        'Conductivity': water.Conductivity,
        'Organic_carbon': water.Organic_carbon,
        'Trihalomethanes': water.Trihalomethanes,
        'Turbidity': water.Turbidity
    
    }, index=[0])
    # Load the RandomForest model once when the application starts

    predicted_value = model.predict(sample)
    # Print result
    if predicted_value == 1:
        return "Water is Consumable"
    else:
        return "Water is not consumable"

