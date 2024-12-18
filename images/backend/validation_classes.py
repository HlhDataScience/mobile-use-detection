"""
This module defines data validation classes using Pydantic for a machine learning application that predicts mobile app usage. It includes classes for input features,
output predictions, query parameters, and API information.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class APIInfo(BaseModel):
    """
    APIInfo

    A class that represents information about the API, including a message,
    description, version, and available endpoints.

    Attributes:
        message (str): A brief message about the API.
        description (str): A detailed description of the API.
        version (float): The version number of the API.
        endpoints (Dict[str, str]): A dictionary mapping endpoint names to their URLs.
    """

    message: str
    description: str
    version: float
    endpoints: Dict[str, str]


class ClassifierInputFeature(BaseModel):
    """
    ClassifierInputFeature

    A class that defines the input features for the classifier model, capturing
    various metrics related to mobile app usage.

    Attributes:
        AppUsageTime_min_day (int): Minutes per day spent using apps on the mobile phone.
        ScreenOnTime_hours_day (float): Hours per day the screen is on.
        BatteryDrain_mAh_day (int): Daily battery usage in mAh.
        NumberOfAppsInstalled (int): The number of apps currently installed on the phone.
        DataUsage_MB_day (int): Daily data usage in megabytes.
    """

    AppUsageTime_min_day: int = Field(
        default=0, description="Minutes per day you use Apps in your mobile phone."
    )
    ScreenOnTime_hours_day: float = Field(
        default=0.0,
        description="Float number of hours you pass with the screen of your phone on.",
    )
    BatteryDrain_mAh_day: int = Field(default=0, description="Usage in mAh per day.")
    NumberOfAppsInstalled: int = Field(
        default=0,
        description="The number of apps you have currently installed in your phone.",
    )
    DataUsage_MB_day: int = Field(
        default=0,
        description="The usage of Data you have currently with your phone.",
    )


class ClassifierOutput(BaseModel):
    """
    ClassifierOutput

    A class that represents the output of the classifier model, including the
    model name, input features, prediction, and timestamp of the inference.

    Attributes:
        ML_model (str): The name of the machine learning model used for prediction.
        features (ClassifierInputFeature): The input features used for the prediction.
        prediction (int): The predicted class by the model, constrained between 1 and 5.
        time_stamp (str): The timestamp when the inference was performed.
    """

    ML_model: str = "Tree_Classifier_New_v4"
    features: ClassifierInputFeature
    prediction: int = Field(
        1, ge=1, le=5, description="The class predicted by the model."
    )
    time_stamp: str = Field("The time you performed the inference.")
    index_id: int = Field(None, description="The actual index in  the Database")


class QueryParameters(BaseModel):
    """
    QueryParameters

    A class that defines the parameters for querying predictions, including
    class constraints, index, and order by criteria.

    Attributes:
        class_ (int): The class to filter predictions, constrained between 1 and 5.
        index (int): The index of the result to retrieve.
        order_by (Literal): The field to order the results by, can be 'index_id',
                            'prediction', or 'time_stamp'.
    """

    class_: int = Field(None, ge=1, le=5)
    index: int = Field(None)
    order_by: Literal["index_id", "prediction", "time_stamp"] = "index_id"


class PredictionHeathers(BaseModel):
    """
    PredictionHeathers

    A class that encapsulates the headers for prediction results, including
    a link to the model metrics.

    Attributes:
        ML_model_metrics (str): A dictionary containing header information
                                   related to the model metrics.
    """

    ML_model_metrics: str = (
        "https://dagshub.com/data_analitics_HLH/mobile-use-detection/models/sk-learn-Decision-Tree-Classifier-custom-params_V4"
    )


class ResultsDisplay(BaseModel):
    """
    ResultsDisplay

    A class that represents the display of results from the classifier,
    including headers and a list of classifier outputs.

    Attributes:
        headers (PredictionHeathers): The headers associated with the prediction results.
        results (List[ClassifierOutput]): A list of classifier output results.
    """

    headers: Optional[PredictionHeathers] = None
    results: List[ClassifierOutput]
