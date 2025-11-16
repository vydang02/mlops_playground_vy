import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from enum import Enum
import mlflow.sklearn
import pandas as pd 
from scripts.session_3.router import predict
from scripts.session_3.router import utils

app = FastAPI()
app.include_router(predict.housing_router) 
app.include_router(utils.utils_router)

@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/health")
def health(dump_input: int):
    if dump_input > 10:
        return {"message": f"This is a large number. Your input is {dump_input}."}
    else:
        return {"message": f"This is a small number. Your input is {dump_input}."}

class Method(str, Enum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"


class CalculateRequest(BaseModel):
    method: Method
    num1: float
    num2: float

class CalculateResponse(BaseModel):
    result: float
    

@app.post("/calculate", response_model=CalculateResponse)
def calculate(request: CalculateRequest) -> CalculateResponse:
    if request.method == Method.add:
        result: float = request.num1 + request.num2
    elif request.method == Method.subtract:
        result: float = request.num1 - request.num2
    elif request.method == Method.multiply:
        result: float = request.num1 * request.num2
    elif request.method == Method.divide:
        result: float = request.num1 / request.num2
    else:
        raise ValueError(f"Invalid method: {request.method}")   
    return CalculateResponse(result=result)

class PredictRequest(BaseModel):
    avg_area_income: float
    avg_area_house_age: float   
    avg_area_number_of_rooms: float
    avg_area_number_of_bedrooms: float
    area_population: float
class PredictResponse(BaseModel):
    predicted_price: float

model_name = "housing_prediction"
model_version = "2"

mlflow.set_tracking_uri("http://127.0.0.1:8080")
model_uri = f"models:/{model_name}/{model_version}"
# model_uri = f"models:/{model_name}@{alias}"

model = mlflow.sklearn.load_model(model_uri)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    data = {
        "Avg. Area Income": [request.avg_area_income],
        "Avg. Area House Age": [request.avg_area_house_age],
        "Avg. Area Number of Rooms": [request.avg_area_number_of_rooms],
        "Avg. Area Number of Bedrooms": [request.avg_area_number_of_bedrooms],
        "Area Population": [request.area_population],
    }
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return PredictResponse(predicted_price=predictions[0])


if __name__ == "__main__":
    uvicorn.run("api_2:app", host="0.0.0.0", port=8000, reload=True)
