# src/predict.py
import joblib # Load trained model 
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def predict_sample():
    # Load trained model 
    # Assumes model.joblib is available in the container's /app directory
    model_path = 'model.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found in container. Exiting.")
        # In a real scenario, you might raise an exception or exit with a non-zero code
        # For this assignment, printing an error and returning might be sufficient for verification.
        return 1 # Return non-zero for script failure

    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}.")

    # Run prediction on test set (or a sample of it) 
    housing = fetch_california_housing(as_frame=True)
    X_test_sample = housing.data.iloc[:5] # Take first 5 samples for prediction

    predictions = model.predict(X_test_sample)

    # Print sample outputs 
    print("\n--- Sample Prediction Test ---")
    print("\nSample Inputs (first 5 features of California Housing data):")
    # Print with more control for better readability if needed
    print(X_test_sample.to_string()) # Use .to_string() for full DataFrame
    print("\nSample Predictions:")
    print(predictions)
    print("\n--- Prediction Test Complete ---")
    return 0 # Return 0 for script success

if __name__ == "__main__":
    exit_code = predict_sample()
    exit(exit_code) # Exit with appropriate code for Docker verification