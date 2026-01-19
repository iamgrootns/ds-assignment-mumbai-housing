import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Defines the function to clean price text (e.g., "22,500" -> 22500.0)
def convert_price_text_to_number(price_text):
    if isinstance(price_text, str):
        # If it's a range like "20,000-22,000", take the average
        if '-' in price_text:
            parts = price_text.split('-')
            try:
                # Remove commas and convert to float
                price_low = float(parts[0].replace(',', '').strip())
                price_high = float(parts[1].replace(',', '').strip())
                return (price_low + price_high) / 2
            except:
                return np.nan
        # If it's a single number
        return float(price_text.replace(',', '').strip())
    # If already a number or empty
    return price_text

def train_property_model():
    print("Loading property data...")
    file_path = "Assignment Data Scientist(in).csv"
    
    try:
        # Load the CSV file
        property_data = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as error:
        print(f"Error loading file: {error}")
        return

    # --- Data Cleaning ---
    # Create a clean 'price_per_sqft' column
    property_data['price_per_sqft'] = property_data['Average Price'].apply(convert_price_text_to_number)
    
    # Remove rows where price is missing
    property_data = property_data.dropna(subset=['price_per_sqft'])
    
    # Extract Year from 'Quarter' (e.g., "Jul-Sep 2024" -> 2024) to use only recent data
    def extract_year(quarter_text):
        if isinstance(quarter_text, str) and ' ' in quarter_text:
            return int(quarter_text.split(' ')[-1])
        return 0
        
    property_data['year'] = property_data['Quarter'].apply(extract_year)
    
    # Filter: Keep only data from 2023 onwards to ensure relevance
    recent_property_data = property_data[property_data['year'] >= 2023].copy()
    
    # If 2023 data is empty (just in case), use all data
    if recent_property_data.empty:
        recent_property_data = property_data.copy()
        
    print(f"Training on {len(recent_property_data)} recent records...")
    
    # --- Feature Engineering & Model Training ---
    
    # Prepare Features (X) and Target (y)
    # We use 'Locality' as the primary feature.
    # Normalize locality names to lowercase for consistency
    recent_property_data['Locality'] = recent_property_data['Locality'].str.lower().str.strip()
    
    X = recent_property_data[['Locality']]
    y = recent_property_data['price_per_sqft']
    
    # Split into Train and Test sets (80% Train, 20% Test)
    # This allows us to validate the model's performance on unseen data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Machine Learning Pipeline
    # 1. OneHotEncoder: Converts categorical 'Locality' into numerical features.
    #    handle_unknown='ignore': If the API sees a new locality, it won't crash (will predict global mean intercept).
    # 2. LinearRegression: Fits a linear model to predict price.
    model_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    print("Training Linear Regression model...")
    model_pipeline.fit(X_train, y_train)
    
    # --- Evaluation ---
    # Predict on the Test set
    y_pred = model_pipeline.predict(X_test)
    
    rmse_error = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_accuracy = r2_score(y_test, y_pred)

    print(f"Model Training Complete.")
    print(f"Root Mean Square Error (RMSE): {rmse_error:.2f}")
    print(f"R2 Score (Accuracy): {r2_accuracy:.2f}")
    
    # --- Save Model ---
    output_filename = 'locality_price_model.pkl'
    with open(output_filename, 'wb') as file:
        pickle.dump(model_pipeline, file)
        
    print(f"Model saved to {output_filename}")

if __name__ == "__main__":
    train_property_model()
