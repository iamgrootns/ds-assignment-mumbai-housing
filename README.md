# Mumbai Property Price Predictor

Hi Manan and Team! This is my submission for the assignment.

This project implements a simple API that predicts property prices in Mumbai based on market trends.

# A Quick Note on the Data
I noticed a small difference between the assignment description and the provided data. The PDF mentioned inputs like `bedrooms` and `furnishing`, but the dataset (`Assignment Data Scientist(in).csv`) actually contained **aggregated price trends** (Price per Sq. Ft.) for each locality, rather than individual apartment listings.

To work around this, I built a **Locality-Based Estimator**. The model looks at the average rate per sq. ft. for a specific area (like "Andheri West") and estimates the total price based on the square footage you provide.

# Project Structure
Here's a quick tour of the files:
- `eda_and_model.ipynb`: The Jupyter Notebook where I explored the data, cleaned it, and built the pricing logic.
- `main.py`: The code for the API (built with FastAPI). It handles the requests and gives you the price.
- `train_model.py`: A script version of the notebook. You can run this to re-train the model anytime.
- `locality_price_model.pkl`: The saved "brain" of the model (a rate card of localities).
- `requirements.txt`: A list of Python libraries needed to run this.

# How to Run It

1. Install the dependencies:
   pip install -r requirements.txt

2. (Optional) Train the model:
   If you want to regenerate the model file, run:
   python train_model.py

3. Start the server:
   uvicorn main:app --reload

# How to Use the API

Once the server is running, you can ask for a prediction!

Send a POST request to `http://127.0.0.1:8000/predict` with data like this:

json
{
    "locality_name": "Andheri West",
    "area_in_sqft": 1000,
    "number_of_bedrooms": 2,
    "number_of_bathrooms": 2,
    "furnishing_status": "Semi-Furnished"
}

# The API will reply with:
{
    "predicted_price": 26262285.71,
    "currency": "INR",
    "price_per_sqft_used": 26262.29,
    "note": "Price estimated using average rates for this locality."
}

Note: The API accepts fields like bedrooms/bathrooms to match the assignment requirements, but currently uses Locality and Area for the calculation.
