import pickle
import numpy as np
import uvicorn
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from typing import Optional


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Welcome to the machine learning API"}

with open(
    r"C:\Users\vykintas.palskys\OneDrive - Thermo Fisher Scientific\Desktop\modul3sprint3\model.pkl", "rb"
) as f:
    model = pickle.load(f)

class InputData(BaseModel):
    passengerid: object
    homeplanet: object
    cryosleep: Optional[bool]
    destination: object
    vip: Optional[bool]
    roomservice: float
    foodcourt: float
    shoppingmall: float
    spa: float
    vrdeck: float
    name: object
    age_cl: object
    cabindeck: object
    cabinnum: int
    cabinside: object

@app.post("/predict/")
def predict(input_data: InputData):
    # input_df = pd.DataFrame([input_data.dict()])
    
    # prediction = model.predict(input_df)

    # return {"prediction": int(prediction[0])}

    input_df = pd.DataFrame([input_data.dict()])
   
   # Ensure that boolean values are correctly handled
    input_df['cryosleep'] = input_df['cryosleep'].astype(bool)
    input_df['vip'] = input_df['vip'].astype(bool)
   
   # Ensure that numeric values are correctly handled
    numeric_columns = ['roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck', 'cabinnum']
    input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
   
   # Ensure that categorical columns are of type str
    categorical_columns = ['passengerid', 'homeplanet', 'destination', 'name', 'age_cl', 'cabindeck', 'cabinside']
    input_df[categorical_columns] = input_df[categorical_columns].astype(str)
   
   # Handle missing values if necessary
    input_df = input_df.fillna('')

    prediction = model.predict(input_df)

    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)