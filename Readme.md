# **MLOps Assignment Report**

**Project Title:** MLOps Pipeline for Predicting California Housing Prices  
 **Dataset:** California Housing Dataset (`sklearn.datasets.fetch_california_housing`)  
 **Model Used:** Linear Regression  
 **Automation Tool:** GitHub Actions  
 **Tools:** Python, scikit-learn, joblib, Conda, Git, GitHub, GitHub Actions

## **Objective**

The objective of this project was to build and automate a complete Machine Learning workflow using MLOps practices. The pipeline was designed to:

* Train a **Linear Regression model** on the California Housing dataset

* Maintain clean and modular code

* Perform automated testing

* Enable inference using a trained model

* Use **GitHub Actions** for Continuous Integration (CI)

## **What is MLOps?**

**MLOps (Machine Learning Operations)** is a practice that combines Machine Learning with DevOps to streamline and automate the lifecycle of ML models. It ensures that models are:

* Reproducible

* Testable

* Automatically trained and deployed

* Easy to monitor and maintain

## **Tools & Technologies**

| Tool | Purpose |
| ----- | ----- |
| Python | Programming language |
| scikit-learn | ML library (model & dataset) |
| joblib | Saving and loading models |
| Git & GitHub | Version control and hosting |
| GitHub Actions | CI for automation |
| pytest | Unit testing framework |
| Conda | Environment and dependency manager |

## **Project Structure**

`housing-regression/`  
`├── src/`  
`│   ├── train.py             # Training logic`  
`│   ├── inference.py         # Inference logic`  
`│   └── utils.py             # Reusable helper functions`  
`├── config/`  
`│   └── config.json          # Hyperparameters & paths`  
`├── tests/`  
`│   └── test_train.py        # Unit tests`  
`├── .github/`  
`│   └── workflows/`  
`│       ├── train.yml        # GitHub Actions CI for training`  
`│       ├── test.yml         # CI for unit testing`  
`│       └── inference.yml    # CI for inference`  
`├── requirements.txt         # Required packages`  
`└── README.md                # Project documentation`

## **Step-by-Step Implementation**

###  **1\. Environment Setup**

A Conda environment was created to manage dependencies:

`conda create -n houseprice python=3.10 -y`  
`conda activate houseprice`

Dependencies were added to `requirements.txt`:

`scikit-learn`  
`joblib`  
`pytest`

Install them via:

`pip install -r requirements.txt`

### **2\. Data Loading**

The dataset used is **California Housing**, fetched using:

python

`from sklearn.datasets import fetch_california_housing`

The dataset contains information about various California districts and median housing prices, used for regression tasks.

The `load_data()` function in `utils.py` handles loading and preparing the dataset for training.

### **3\. Model Training (`train.py`)**

* Loads data using `utils.py`

* Reads hyperparameters from `config/config.json`

* Trains a **Linear Regression** model

* Saves the model to disk using `joblib` as `model.pkl`

###  **4\. Configuration File (`config/config.json`)**

Used to store parameters such as:

json

`{`  
  `"fit_intercept": true,`  
  `"normalize": false`  
`}`

This allows changing model parameters without modifying code.

###  **5\. Model Saving & Loading (via `joblib`)**

* The trained model is saved as `model.pkl` using:

python

`joblib.dump(model, 'model.pkl')`

* The model is later loaded in `inference.py` for prediction.

### 

###  **6\. Inference Script (`inference.py`)**

This script:

* Loads the saved model

* Loads a sample input (can be synthetic or from dataset)

* Prints the predicted house price

###  **7\. Unit Testing (`test_train.py`)**

* Verifies model training runs without errors

* Confirms model file `model.pkl` is created

* Validates model type (instance of LinearRegression)

###  **8\. GitHub Actions Workflows**

CI was implemented using **GitHub Actions** under `.github/workflows/`.

####  **`train.yml`**

* Triggers on push

* Installs dependencies

* Runs `train.py`

* Uploads model as artifact

####  **`test.yml`**

* Runs `pytest` on `test_train.py`

* Ensures training script is functioning correctly

####  **`inference.yml`**

* Downloads trained model artifact

* Runs `inference.py`

* Outputs predicted results in CI logs

##  **Branching Strategy**

A simple linear branch flow was used:

`main ➔ train ➔ test ➔ inference`

Each branch was responsible for a specific stage in the MLOps pipeline.

## **Model Evaluation**

Since this is a regression problem, the model can be evaluated using:

* **Mean Squared Error (MSE)**

* **R² Score**

Metrics can be printed during training and logged in CI if needed.

## **Key Takeaways**

* Set up an ML pipeline using modular code

* Implemented CI with GitHub Actions

* Learned how to manage dependencies and configurations

* Created a reproducible and automated workflow

* Understood version control through branching

##  **Deliverables**

| File / Folder | Description |
| ----- | ----- |
| `src/train.py` | Training logic |
| `src/inference.py` | Inference logic |
| `src/utils.py` | Reusable functions |
| `config/config.json` | Model configurations |
| `tests/test_train.py` | Unit tests |
| `.github/workflows/*.yml` | GitHub CI pipelines |
| `model.pkl` (artifact) | Saved trained model |
| `requirements.txt` | Python dependencies |
| `README.md` | Documentation and usage instructions |

