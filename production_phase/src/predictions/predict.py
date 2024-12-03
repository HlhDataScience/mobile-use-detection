"""Provisional module"""

import joblib
from pydantic import BaseModel


# Define the input schema for predictions
class PredictionInput(BaseModel):
    """small pydantic class to be further implemented"""

    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float


# Define the prediction endpoint logic
def predict_endpoint(input_data: dict, classifier) -> dict:
    """function that handles the predictions of the ML model"""
    # Extract features from input_data
    features = [
        [
            input_data["feature1"],
            input_data["feature2"],
            input_data["feature3"],
            input_data["feature4"],
            input_data["feature5"],
        ]
    ]
    # Make a prediction using the classifier
    predicted_class = classifier.predict(features)[0]
    return {"predicted_class": int(predicted_class)}


def load_classifier(model_path: str):
    """Loads the model for the predictions"""
    with open(model_path, "rb") as f:
        return joblib.load(f)
