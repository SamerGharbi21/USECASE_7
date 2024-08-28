from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

model = joblib.load('DBSCAN.joblib')
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
    try:
        # Preprocess input data
        data = preprocessing(input_features)
        # Predict the cluster using the DBSCAN model
        y_pred = model.fit_predict(data)  # Use fit_predict for DBSCAN
        return {"cluster": int(y_pred[0])}
    except Exception as e:
        # Return error message if any exception occurs
        raise HTTPException(status_code=400, detail=str(e))

