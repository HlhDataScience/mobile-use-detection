"""Setup module that may be expanded with other frameworks."""

import json
from datetime import datetime, timezone
from typing import Annotated, Dict

import joblib  # type: ignore
import numpy as np
from fastapi import Header, HTTPException, Query, Request

from src.production.backend.validation_classes import (
    APIInfo,
    ClassifierInputFeature,
    ClassifierOutput,
    PredictionHeathers,
    QueryParameters,
    ResultsDisplay,
)

PREDICTION_FILE = "predictions.json"


async def read_root() -> APIInfo:
    """presentation of the api"""
    return APIInfo(
        **{
            "message": "Welcome to the mobile-use-detection API!",
            "description": "This API allows you to make predictions and consult results related to the use of mobile app you do in your daily life.",
            "version": 1.0,
            "endpoints": {
                "/predict": "Make a prediction.",
                "/results": "Consult all results.",
                "/results/get_results": "Filter the results",
                "/docs": "API documentation (Swagger UI).",
                "/openapi.json": "OpenAPI schema.",
            },
            "metadata": {
                "Github repository": "https://github.com/HlhDataScience/mobile-use-detection",
                "Dagshug repository": "https://dagshub.com/data_analitics_HLH/mobile-use-detection",
            },
        }
    )


def load_classifier(model_path: str):
    """Loads the ModelsProduction for the predictions."""
    with open(model_path, "rb") as f:
        return joblib.load(f)


def save_results(prediction_entry: Dict) -> None:
    """Save the predictions and results into a JSON file."""
    try:
        with open(f"src/production/backend/{PREDICTION_FILE}", "r") as f:
            predictions = json.load(f)
    except FileNotFoundError:
        predictions = []  # If file doesn't exist, start with an empty list

    predictions.append(prediction_entry)
    for index, dicti in enumerate(predictions):
        dicti["index_id"] = index
    with open(f"src/production/backend/{PREDICTION_FILE}", "w") as f:
        json.dump(predictions, f, indent=4)  # type: ignore


async def classify(
    input_data: ClassifierInputFeature, request: Request
) -> ClassifierOutput:
    """Classifies the results sent by users into classes of mobile usage."""
    try:
        classifier = request.app.state.classifier

        input_data_list = [i for i in input_data.model_dump().values()]
        numbers = np.array(input_data_list).reshape(1, -1)

        predicted_class = classifier.predict(numbers)[0]  # Get the predicted class
        prediction = int(predicted_class)
        validated_entry = ClassifierOutput(
            features={k: v for k, v in input_data.model_dump().items()},
            prediction=prediction,
            time_stamp=datetime.now(timezone.utc).isoformat(),
            index_id=0,
        )

        save_results(prediction_entry=validated_entry.model_dump())
        # Return the result
        return ClassifierOutput(
            features={k: v for k, v in input_data.model_dump().items()},
            prediction=prediction,
            time_stamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


async def results(headers: Annotated[PredictionHeathers, Header()]) -> ResultsDisplay:
    with open(PREDICTION_FILE, "r") as f:
        json_file = json.load(f)

    return ResultsDisplay(
        results=json_file, headers={k: v for k, v in headers.model_dump().items()}
    )


async def get_results(
    headers: Annotated[PredictionHeathers, Header()],
    filter_query: Annotated[QueryParameters, Query()],
) -> ResultsDisplay:
    """Obtains the storage results of previous predictions."""

    with open(PREDICTION_FILE, "r") as f:
        json_file = json.load(f)
    try:
        headers_var = {k: v for k, v in headers.model_dump().items()}
        filtered_results = iter(json_file)
        if filter_query.class_ is None and filter_query.index is None:
            return ResultsDisplay(
                results=json_file,
                headers=headers_var,
            )
        # Apply filtering
        filtered_results = (
            entry
            for entry in filtered_results
            if (
                filter_query.class_ is None
                or entry.get("prediction") == filter_query.class_
            )
            and (
                filter_query.index is None
                or entry.get("index_id") == filter_query.index
            )
        )

        # Sort the results
        filtered_results = sorted(
            filtered_results, key=lambda x: x.get(filter_query.order_by, 0)
        )

        # Check if there are any results
        if not filtered_results:
            raise HTTPException(status_code=404, detail="No match found")

        return ResultsDisplay(
            results=filtered_results,
            headers={k: v for k, v in headers.model_dump().items()},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")


"""
POSSIBLE IMPLEMENTATION FOR BINARY TREE SEARCH INSIDE THE API FOR JSON FILE.
import json
from bisect import bisect_left
from fastapi import HTTPException
from typing import List, Dict, Any

async def search_results(json_file: List[Dict[str, Any]], filter_query: QueryParameters) -> List[Dict[str, Any]]:
    # Sort the json_file by index_id and class_ for binary search
    json_file.sort(key=lambda x: (x.get("index_id", 0), x.get("prediction", 0)))
        filtered_results = []
    # Perform binary search for index_id
    if filter_query.index is not None:
        index_list = [entry["index_id"] for entry in json_file]
        pos = bisect_left(index_list, filter_query.index)
        # Collect results that match the index_id
        while pos < len(json_file) and json_file[pos]["index_id"] == filter_query.index:
            filtered_results.append(json_file[pos])
            pos += 1
    # Perform binary search for class_ if specified
    if filter_query.class_ is not None:
        # Filter the results based on class_ using binary search
        class_list = [entry["prediction"] for entry in filtered_results]
        class_pos = bisect_left(class_list, filter_query.class_)
        # Collect results that match the class_
        filtered_results = [
            entry for entry in filtered_results[class_pos:]
            if entry["prediction"] == filter_query.class_
        ]

    # Sort the results if needed
    if filter_query.order_by:
        filtered_results.sort(key=lambda x: x.get(filter_query.order_by, 0))

    return filtered_results

async def get_results(
    headers: Annotated[PredictionHeathers, Header()],
    filter_query: Annotated[QueryParameters, Query()],
) -> ResultsDisplay:

    with open(PREDICTION_FILE, "r") as f:
        json_file = json.load(f)

    try:
        headers_var = {k: v for k, v in headers.model_dump().items()}

        if filter_query.class_ is None and filter_query.index is None:
            return ResultsDisplay(
                results=json_file,
                headers=headers_var,
            )

        # Call the search_results function and await its results
        filtered_results = await search_results(json_file, filter_query)

        # Check if there are any results
        if not filtered_results:
            raise HTTPException(status_code=404, detail="No match found")

        return ResultsDisplay(
            results=filtered_results,
            headers=headers_var,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{str(e)}")
        """
