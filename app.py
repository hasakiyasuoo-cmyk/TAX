from fastapi import FastAPI
from pydantic import BaseModel
from backend.tax_model import TaxPredictor
from backend.optimizer import TaxOptimizer

app = FastAPI(title="智税未来管家 API")

predictor = TaxPredictor()
optimizer = TaxOptimizer()

class TaxRequest(BaseModel):
    income: float
    expenses: float
    deductions: float

@app.post("/predict")
def predict_tax(request: TaxRequest):
    prediction = predictor.predict(income=request.income, expenses=request.expenses, deductions=request.deductions)
    return {"predicted_tax": prediction}

@app.post("/optimize")
def optimize_tax(request: TaxRequest):
    plan = optimizer.optimize(request.income, request.expenses, request.deductions)
    return {"optimized_plan": plan}