"""Setup module that may be expanded with other frameworks."""

from typing import Dict

from fastapi import Request

from production_phase.src.interfaces.WebFrameworksProtocols import (
    WebFrameworkProtocol,
)
from production_phase.src.predictions.predict import (
    PredictionInput,
    predict_endpoint,
)


async def root() -> Dict:
    return {"message": "THis is a test of the interface"}


async def read_item(item_id: int) -> Dict:
    return {"item_id": item_id}


def setup_app(
    framework: WebFrameworkProtocol,
    classifier,
):
    async def prediction_wrapper(request: Request):
        """Asyncronous function to  wrapper the predictions"""
        # Extract the JSON body
        input_data = await request.json()
        # Validate and process the input
        validated_data = PredictionInput(**input_data)
        # Pass validated data to the prediction endpoint
        return predict_endpoint(validated_data.model_dump(), classifier)

    # Add the prediction route to the framework
    framework.add_route("/predict", prediction_wrapper, methods=["POST"])
