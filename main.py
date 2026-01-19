from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import os
import pandas as pd
import numpy as np

# Initialize the API Application
app = FastAPI(title="Mumbai Property Price Estimator", version="2.0")

# --- Definitions ---

class PropertyDetails(BaseModel):
    locality: str
    area_sqft: float = Field(..., gt=0, description="Area must be positive")
    bedrooms: int = Field(..., ge=0, description="Cannot have negative bedrooms")
    bathrooms: int = Field(..., ge=0, description="Cannot have negative bathrooms")
    furnishing: str

# --- Load the Model ---

model_filename = "locality_price_model.pkl"
model = None

if os.path.exists(model_filename):
    with open(model_filename, "rb") as file:
        model = pickle.load(file)
    print("ML Model loaded successfully.")
else:
    print("WARNING: Model file not found. Please run 'train_model.py' first.")

#API Endpoints

@app.get("/")
def home_page():
    return {"message": "Welcome to the Mumbai Property Price Estimator API (v2.0)"}

@app.post("/predict")
def calculate_property_price(property: PropertyDetails):
    """
    Predicts the price of a property using a trained Linear Regression model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please contact administrator.")

    # 1. Preprocess Input
    # Normalize locality to match training data (lowercase)
    cleaned_locality = property.locality.strip().lower()
    
    # Create a DataFrame for the model (Pipeline expects a DataFrame with 'Locality' column)
    input_data = pd.DataFrame({'Locality': [cleaned_locality]})
    
    # 2. Predict Price per Sqft
    try:
        predicted_price_per_sqft = model.predict(input_data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
    # Handle negative predictions (unlikely but possible with linear regression)
    if predicted_price_per_sqft < 0:
        predicted_price_per_sqft = 0.0

    # 3. Calculate Total Price
    total_estimated_price = predicted_price_per_sqft * property.area_sqft
    
    # 4. Return Result
    return {
        "predicted_price": round(total_estimated_price, 2),
        "currency": "INR",
        "price_per_sqft_used": round(predicted_price_per_sqft, 2),
        "locality_used": cleaned_locality,
        "note": "Price predicted using Linear Regression model on recent market trends."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
