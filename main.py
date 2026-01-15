from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

# Initialize the API Application
app = FastAPI(title="Mumbai Property Price Estimator", version="1.0")

#Definitions

# Define the data we expect from the user
from pydantic import BaseModel, Field

# Define the data we expect from the user
class PropertyDetails(BaseModel):
    locality_name: str
    area_in_sqft: float = Field(..., gt=0, description="Area must be positive")
    number_of_bedrooms: int = Field(..., ge=0, description="Cannot have negative bedrooms")
    number_of_bathrooms: int = Field(..., ge=0, description="Cannot have negative bathrooms")
    furnishing_status: str

#Load the Model

model_filename = "locality_price_model.pkl"
locality_average_prices = {}

#Check if the model file exists, then load it
if os.path.exists(model_filename):
    with open(model_filename, "rb") as file:
        locality_average_prices = pickle.load(file)
    print(f"Model loaded successfully. Knowing prices for {len(locality_average_prices)} localities.")
else:
    print("WARNING: Model file not found. Please run 'train_model.py' first.")

#API Endpoints

@app.get("/")
def home_page():
    return {"message": "Welcome to the Mumbai Property Price Estimator API"}

@app.post("/predict")
def calculate_property_price(property: PropertyDetails):
    """
    Predicts the price of a property based on its Location and Area.
    """
    
    #Get the requested locality
    requested_locality = property.locality_name
    
    #Find the price per sqft for this locality in our database
    price_per_sqft = 0.0
    
    # Try exact match
    if requested_locality in locality_average_prices:
        price_per_sqft = locality_average_prices[requested_locality]
    else:
        # Try case-insensitive match (e.g. "andheri" matches "Andheri")
        found = False
        for known_locality in locality_average_prices:
            if known_locality.lower() == requested_locality.lower():
                price_per_sqft = locality_average_prices[known_locality]
                found = True
                break
        
        if not found:
            raise HTTPException(status_code=404, detail=f"Locality '{requested_locality}' not found in our database.")

    #Calculate Total Price = Rate * Area
    total_estimated_price = price_per_sqft * property.area_in_sqft
    
    #Return the result
    return {
        "predicted_price": round(total_estimated_price, 2),
        "currency": "INR",
        "price_per_sqft_used": round(price_per_sqft, 2),
        "note": "Price estimated using average rates for this locality."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
