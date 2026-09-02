# Containerized ML Prediction API

A small end-to-end machine learning deployment project demonstrating how a trained scikit-learn model can be served as a REST API and containerized with Docker.

## Overview

This project trains a Support Vector Machine (SVM) classifier on the **Breast Cancer Wisconsin Diagnostic Dataset** and exposes the trained model through a FastAPI application.

The project focuses on the transition from a trained ML model to a deployable inference service rather than on complex model development.

### Pipeline

```text
Dataset
   ↓
Train / Test Split
   ↓
StandardScaler + SVM
   ↓
Model Evaluation
   ↓
Joblib Serialization
   ↓
FastAPI
   ↓
Docker Container
   ↓
REST API
```

## Technologies

* Python 3.11
* scikit-learn
* NumPy
* FastAPI
* Pydantic
* Uvicorn
* Joblib
* Docker

## Model

The model uses an SVM classifier preceded by feature standardization:

```text
StandardScaler → SVC
```

The preprocessing and classifier are stored together as a scikit-learn `Pipeline` and serialized with Joblib.

The dataset contains **569 samples and 30 numerical features** describing characteristics of cell nuclei extracted from digitized breast mass images.

The data is split into:

* 80% training data
* 20% test data

The resulting model achieved approximately **98.25% accuracy on the held-out test set**.

## API

The FastAPI application provides two endpoints.

### Health check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Prediction

```http
POST /api/predict
```

Request:

```json
{
  "features": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}
```

Response:

```json
{
  "prediction": 0,
  "probability": 0.7646682128299337
}
```

The API validates the request using Pydantic and converts the feature list into the format expected by the trained model.

## Interactive API Documentation

FastAPI automatically generates interactive Swagger documentation.

When running locally, it is available at:

```text
http://localhost:8000/docs
```

This can be used to send test requests directly to the API.

## Running with Docker

Build the Docker image:

```bash
docker build -t ml-inference-api .
```

Run the container:

```bash
docker run -p 8000:8000 ml-inference-api
```

The API is then available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Project Structure

```text
ml_inference_api/
│
├── app/
│   └── main.py
│
├── model/
│   └── svm_classification.joblib
│
├── train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## What This Project Demonstrates

* Training a supervised ML model with scikit-learn
* Building preprocessing and inference into a reusable pipeline
* Persisting a trained model with Joblib
* Loading a trained model without retraining
* Creating a REST-style inference API with FastAPI
* Request validation with Pydantic
* Returning model predictions and probabilities as JSON
* Generating interactive OpenAPI/Swagger documentation
* Containerizing an ML application with Docker
* Running an ML inference service independently of the host Python environment

## Future Improvements

Possible extensions include:

* Input validation for the expected 30 features and their ranges
* Automated tests for API endpoints
* Docker health checks
* Environment-based model configuration
* CI/CD with GitHub Actions
* Model versioning
* API authentication
* Deployment to a cloud platform

## Disclaimer

This project is intended as a demonstration of machine learning deployment and software engineering practices. It is **not a medical diagnostic system** and should not be used for clinical decision-making.
