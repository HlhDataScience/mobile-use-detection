"""THis module houses the functions for inference of the fastapi and wrappers for gradio and gradio blocks."""

from typing import Dict

import requests


def inference_point(input_data: Dict[str, int | float]) -> str | Dict[str, str]:
    """
    Sends a POST request to an API for inference.

    Parameters:
    - api_url (str): The API endpoint URL.
    - input_data (dict): The input data for inference.

    Returns:
    - dict: The API response in JSON format.
    """
    api_endpoint = "http://127.0.0.1:8001/predict/"
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    try:
        response = requests.post(api_endpoint, json=input_data, headers=headers)
        response.raise_for_status()
        human_response = response.json()
        return f"Your usage of your smartphone is classified as class {human_response['prediction']}"
    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors
        if e.response is not None:
            try:
                error_response = e.response.json()  # Try to get the JSON response
                return {"error": error_response}  # Return the detailed error response
            except ValueError:
                return {
                    "error": str(e)
                }  # If the response is not JSON, return the error message
        else:
            return {"error": str(e)}  # If there's no response, return the error message
    except Exception as e:
        return {"error": str(e)}  # Handle other exceptions


def gradio_inference(
    AppUsageTime: int,
    ScreenOnTime: float,
    BatteryDrain: int,
    NumApps: int,
    DataUsage: int,
):
    input_features = {
        "AppUsageTime_min_day": AppUsageTime,
        "ScreenOnTime_hours_day": ScreenOnTime,
        "BatteryDrain_mAh_day": BatteryDrain,
        "NumberOfAppsInstalled": NumApps,
        "DataUsage_MB_day": DataUsage,
    }
    return inference_point(input_features)


def provide_class_info(class_prediction: str) -> str:
    """
    Provides additional information based on the predicted class.

    Parameters:
    - class_prediction (str): The class prediction output from `inference_point`.

    Returns:
    - str: Additional information about the predicted class.
    """
    class_mapping = {
        "class 1": "You have minimal smartphone usage. Great job maintaining balance!",
        "class 2": "Your smartphone usage is moderate. Consider tracking screen time.",
        "class 3": "Your usage is on the higher side. Taking breaks might help.",
        "class 4": "You use your smartphone extensively. Watch out for digital fatigue.",
        "class 5": "Excessive usage detected! It's time to reassess your habits.",
    }

    # Extract the class number from the string (e.g., "class 0")
    for key, value in class_mapping.items():
        if key in class_prediction:
            return value

    return "No additional information available for the predicted class."
