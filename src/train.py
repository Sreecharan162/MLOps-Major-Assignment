# src/train.py
import joblib
from sklearn.datasets import fetch_california_housing # Use California Housing dataset 
from sklearn.linear_model import LinearRegression # Use scikit-learn's LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os # Import os for path handling

def train_model():
    # Load the California Housing dataset 
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LinearRegression model 
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print R^2 score and loss (Mean Squared Error) 
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    loss = mean_squared_error(y_test, y_pred)

    print(f"R^2 Score: {r2}")
    print(f"Loss (Mean Squared Error): {loss}")

    # Save model using joblib 
    # Save to the root directory as per Dockerfile copying
    model_path = 'model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {os.path.abspath(model_path)}") # Print absolute path for clarity

    return r2, loss, model_path

if __name__ == "__main__":
    train_model()