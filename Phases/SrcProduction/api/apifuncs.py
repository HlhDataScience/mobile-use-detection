"""Setup module that may be expanded with other frameworks."""

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib  # type: ignore
import numpy as np
from fastapi import HTTPException, Request

from Phases.SrcProduction.api.validation_classes import ClassifierInputFeature
from Phases.SrcProduction.interfaces.WebFrameworksProtocols import (
    EndPointProtocolFunction,
    WebFrameworkProtocol,
)

PREDICTION_FILE = "predictions.json"


def save_results(prediction_entry) -> None:
    """Save the predictions and results into a JSON file"""

    try:
        with open(
            f"production_phase/data/{PREDICTION_FILE}", "r"
        ) as f:  # Added comma here
            predictions = json.load(f)
    except FileNotFoundError:
        predictions = []  # If file doesn't exist, start with an empty list

    predictions.append(prediction_entry)
    for index, dicti in enumerate(predictions):
        if "index_id" not in dicti.keys():
            dicti["index_id"] = index

    # Write the updated DataTrain back to the JSON file
    with open(f"production_phase/data/{PREDICTION_FILE}", "w") as f:
        json.dump(predictions, f, indent=4)  # type: ignore


async def get_results(
    index_id: int, query: str | None = None, parameters: str | None = None
) -> Dict | str | int | float:
    """obtains the storage results of previous predictions."""
    json_file = await results()
    try:
        if not query:
            result = dict(
                next(
                    filter(
                        lambda dicti: dicti["index_id"] == index_id,
                        json_file["results"],
                    ),
                    "Index not present",
                )
            )  # type: ignore
        else:
            if not parameters:
                result = dict(
                    next(
                        filter(
                            lambda dicti: dicti["index_id"] == index_id,
                            json_file["results"],
                        ),
                        "Index not present",
                    )
                )  # type: ignore
                if isinstance(result, dict):
                    result = result[query]
                else:
                    result = result.items()
            else:
                result = dict(
                    next(
                        filter(
                            lambda dicti: dicti["index_id"] == index_id,
                            json_file["results"],
                        ),
                        "Index not present",
                    )
                )  # type: ignore

                result = result[query]
                if isinstance(result, dict):
                    result = result[parameters]
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


async def results() -> Dict:
    with open(f"production_phase/data/{PREDICTION_FILE}", "r") as f:
        json_file = json.load(f)
    return {"results": json_file}


def load_classifier(model_path: str):
    """Loads the ModelsProduction for the predictions"""
    with open(model_path, "rb") as f:
        return joblib.load(f)


async def root() -> Dict:
    """small testing root function"""
    return {"message": "This is a test of the interface"}


async def classify(input_data: ClassifierInputFeature, request: Request):
    """Classifies the results sent by users into classes of mobile usage."""
    try:
        classifier = request.app.state.classifier
        # Extract and reshape the input DataTrain
        input_data_list = [i for i in input_data.model_dump().values()]
        numbers = np.array(input_data_list).reshape(1, -1)
        # Make prediction using the tree classifier
        predicted_class = classifier.predict(numbers)[0]  # Get the predicted class
        prediction = {"class": int(predicted_class)}
        predicted_entry = {
            "features": input_data.model_dump(),
            "prediction": prediction,
            "time_stamp": datetime.now(timezone.utc).isoformat(),
        }
        save_results(prediction_entry=predicted_entry)
        # Return the result
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def setup_app(
    framework: WebFrameworkProtocol,
    api_functions: Dict[str, Tuple[EndPointProtocolFunction, List[str]]],
) -> WebFrameworkProtocol:
    """Creates the API setup from a dictionary"""
    for path, parameters in api_functions.items():
        framework.add_route(path=path, endpoint=parameters[0], methods=parameters[1])

    return framework
