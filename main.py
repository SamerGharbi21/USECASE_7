from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

model = joblib.load('DBSCAN1.joblib')
scaler = joblib.load('DBscaler.joblib')

app = FastAPI()

class InputFeatures(BaseModel):
    age: int
    current_value: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'age': input_features.age,
        'current_value': input_features.current_value,
    }
    feature_list = [dict_f[key] for key in sorted(dict_f)]
    return scaler.transform([feature_list])

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/try/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/predict")
async def predict(input_features: InputFeatures):
        data = preprocessing(input_features)
        y_pred = model.fit_predict(data)  
        return {"cluster": int(y_pred[0])}


