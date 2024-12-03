"""Setup module that may be expanded with other frameworks."""

from typing import Any, Callable, Dict, List, Literal, Tuple

from fastapi import Request

from production_phase.src.interfaces.WebFrameworksProtocols import (
    WebFrameworkProtocol,
)
from production_phase.src.predictions.predict import (
    PredictionInput,
    predict_endpoint,
)


async def root() -> Dict:
    """small testing root function"""
    return {"message": "THis is a test of the interface"}


async def read_item(item_id: int) -> Dict:
    """small testinmg function"""
    return {"item_id": item_id}


async def prediction_wrapper(classifier, request: Request):
    """Asynchronous function to  wrapper the predictions"""
    # Extract the JSON body
    input_data = await request.json()

    # Validate and process the input
    validated_data = PredictionInput(**input_data)
    # Pass validated data to the prediction endpoint
    return predict_endpoint(validated_data.model_dump(), classifier)


def setup_app(
    framework: WebFrameworkProtocol,
    api_functions: Dict,
):
    """Creates the API setup from a dictionary"""
    for path, parameters in api_functions.items():
        framework.add_route(path=path, endpoint=parameters[0], methods=parameters[1])

    return framework
