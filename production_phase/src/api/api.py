"""Setup module that may be expanded with other frameworks."""

from fastapi import Request

from production_phase.src.interfaces.WebFrameworksProtocols import (
    WebFrameworkProtocol,
)
from production_phase.src.predictions.predict import (
    PredictionInput,
    predict_endpoint,
)


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
