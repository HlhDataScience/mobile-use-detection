"""Setup module that may be expanded with other frameworks."""

from typing import Dict

import joblib  # type: ignore
import numpy as np
from fastapi import FastAPI, HTTPException

from EDA_train_phase.src.validation_classes.validation_interfaces import (
    PydanticConfigModel,
)
from production_phase.src.api.validation_classes import ClassifierInputFeature
from production_phase.src.interfaces.WebFrameworksProtocols import (
    WebFrameworkProtocol,
)

pydantic_model = PydanticConfigModel(config_model=ClassifierInputFeature)
# CONSTANTS:
MODEL_PATH = "EDA_train_phase/models/Tree_Classifier_New_v4.joblib"


def load_classifier(model_path: str):
    """Loads the model for the predictions"""
    with open(model_path, "rb") as f:
        return joblib.load(f)


async def root() -> Dict:
    """small testing root function"""
    return {"message": "This is a test of the interface"}


async def read_item(item_id: int) -> Dict:
    """small testing function"""
    return {"item_id": item_id}


async def classify(input_data: ClassifierInputFeature):
    try:
        classifier = load_classifier(model_path=MODEL_PATH)
        # Extract and reshape the input data
        input_data_list = [i for i in input_data.model_dump().values()]
        numbers = np.array(input_data_list).reshape(1, -1)
        # Make prediction using the tree classifier
        predicted_class = classifier.predict(numbers)[0]  # Get the predicted class

        # Return the result
        return {"class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def setup_app(
    framework: WebFrameworkProtocol,
    api_functions: Dict,
) -> WebFrameworkProtocol:
    """Creates the API setup from a dictionary"""
    for path, parameters in api_functions.items():
        framework.add_route(path=path, endpoint=parameters[0], methods=parameters[1])

    return framework
