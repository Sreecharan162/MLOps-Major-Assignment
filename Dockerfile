# Dockerfile 
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your source code (src/ folder)
COPY src/ ./src/

# Copy the trained model and quantized parameters
# These will be created by the train_and_quantize job and downloaded by build_and_test_container
COPY model.joblib .
COPY unquant_params.joblib .
COPY quant_params.joblib .

# Command to run when the container starts.
# The CI/CD workflow will specifically call 'python src/predict.py'.
# This CMD is a default if no command is specified when running the container.
CMD ["python", "src/predict.py"]