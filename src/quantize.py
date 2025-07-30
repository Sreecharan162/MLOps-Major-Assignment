# src/quantize.py
import joblib # Load trained model 
import numpy as np
import os

def quantize_model(model_path='model.joblib'):
    # Load trained model 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    # Extract coef and intercept 
    coef = model.coef_
    intercept = model.intercept_
    print(f"Extracted coefficients (shape: {coef.shape}) and intercept: {intercept}")

    # Save raw parameters (unquant_params.joblib) 
    unquant_params = {'coef': coef, 'intercept': intercept}
    joblib.dump(unquant_params, 'unquant_params.joblib')
    print(f"Unquantized parameters saved to unquant_params.joblib")

    # Manually quantize them to unsigned 8-bit integers 
    # This is a basic linear quantization for demonstration.
    # Find min/max values across both coef and intercept for consistent scaling
    all_params = np.concatenate((coef.flatten(), [intercept]))
    min_val = np.min(all_params)
    max_val = np.max(all_params)

    # Calculate scale and zero_point for 8-bit unsigned quantization (0-255)
    # Avoid division by zero if all params are identical
    if max_val == min_val:
        scale = 1.0 if min_val != 0 else 1.0/255.0 # Handle cases where all values are the same
        zero_point = 0
    else:
        scale = (max_val - min_val) / 255.0
        # Zero-point is the quantized value corresponding to the original float 0.0
        # Here, we map original_min to 0 and original_max to 255
        zero_point = int(np.round(0 - (min_val / scale)))

    # Clamp zero_point to be within the 0-255 range
    zero_point = np.clip(zero_point, 0, 255)

    # Quantize coefficients and intercept
    quantized_coef = np.round((coef / scale) + zero_point).astype(np.uint8)
    quantized_intercept = np.round((intercept / scale) + zero_point).astype(np.uint8)
    print(f"Quantized coefficients (shape: {quantized_coef.shape}) and intercept: {quantized_intercept}")


    # Save quantized parameters (quant_params.joblib) 
    quant_params = {
        'coef': quantized_coef,
        'intercept': quantized_intercept,
        'scale': scale,
        'zero_point': zero_point,
        'original_min': min_val,
        'original_max': max_val
    }
    joblib.dump(quant_params, 'quant_params.joblib')
    print(f"Quantized parameters saved to quant_params.joblib")

    # Perform inference with the de-quantized weights 
    # To do this, we need to convert them back to float
    dequantized_coef = (quantized_coef.astype(np.float32) - zero_point) * scale
    dequantized_intercept = (quantized_intercept.astype(np.float32) - zero_point) * scale

    # For demonstration, create a simple class that mimics LinearRegression's predict method
    class DeQuantizedLinearRegression:
        def __init__(self, coef, intercept):
            self.coef_ = coef
            self.intercept_ = intercept

        def predict(self, X):
            # Ensure X is a numpy array for dot product, especially important if X comes from pandas
            if isinstance(X, np.ndarray):
                return X @ self.coef_ + self.intercept_
            else: # Assume it's a pandas DataFrame or similar, convert to numpy
                return X.values @ self.coef_ + self.intercept_

    dequantized_model = DeQuantizedLinearRegression(dequantized_coef, dequantized_intercept)
    print("De-quantized parameters ready for inference simulation.")
    return dequantized_model


if __name__ == "__main__":
    # First, ensure model.joblib exists by running train.py
    # This is for local testing. In CI/CD, train_and_quantize job will handle it.
    from src.train import train_model
    print("Ensuring model.joblib is present by running training...")
    train_model()
    print("-" * 30)

    # Then, run quantization
    dequantized_model = quantize_model()
    print("-" * 30)

    # Optional: Compare predictions
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing(as_frame=True)
    X_test_sample = housing.data.iloc[:5] # Get a few samples for testing

    print("\nOriginal model prediction (sample):")
    original_model = joblib.load('model.joblib')
    print(original_model.predict(X_test_sample))

    print("\nDe-quantized model prediction (sample):")
    print(dequantized_model.predict(X_test_sample))

    # Clean up files created for local testing
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('unquant_params.joblib'):
        os.remove('unquant_params.joblib')
    if os.path.exists('quant_params.joblib'):
        os.remove('quant_params.joblib')
    print("\nCleaned up locally generated model and parameter files.")