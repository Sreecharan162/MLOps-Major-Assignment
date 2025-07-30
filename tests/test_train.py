# tests/test_train.py
import joblib
import os
from sklearn.datasets import fetch_california_housing # Unit test dataset loading 
from sklearn.linear_model import LinearRegression # Validate model creation 
from src.train import train_model # Import the training function

# Define a minimum acceptable R^2 score 
MIN_R2_THRESHOLD = 0.5 # A reasonable starting point for this dataset

def test_dataset_loading():
    # Unit test dataset loading 
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    assert X is not None, "Dataset features X should not be None."
    assert y is not None, "Dataset target y should not be None."
    assert len(X) > 0, "Dataset features X should not be empty."
    assert len(y) > 0, "Dataset target y should not be empty."
    print("Test passed: Dataset loaded successfully.")

def test_model_creation():
    # Validate model creation (LinearRegression instance) 
    model = LinearRegression()
    assert isinstance(model, LinearRegression), "Model should be an instance of LinearRegression."
    print("Test passed: LinearRegression model instance created.")

def test_model_training():
    # Train the model and get results
    r2, loss, model_path = train_model()

    # Check if model file exists 
    assert os.path.exists(model_path), f"Model file '{model_path}' was not created."

    # Load the trained model to check its attributes 
    model = joblib.load(model_path)
    assert hasattr(model, 'coef_'), "Trained model should have 'coef_' attribute."
    assert hasattr(model, 'intercept_'), "Trained model should have 'intercept_' attribute."
    print("Test passed: Model trained and has coefficients/intercept.")

    # Ensure R^2 score exceeds minimum threshold 
    assert r2 > MIN_R2_THRESHOLD, f"R^2 score {r2} is below threshold {MIN_R2_THRESHOLD}."
    print(f"Test passed: R^2 score {r2} exceeds minimum threshold {MIN_R2_THRESHOLD}.")

    # Clean up the generated model file after testing
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleaned up: Removed {model_path}")