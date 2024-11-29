"""
This module contains classes and functions for data validation. The module includes the following key components:

- DataValidationConfig: A Pandera-based configuration class for validating incoming data and ensuring it adheres to an expected schema.
- DataTransformationConfig: A Pydantic-based configuration class for managing transformation settings, including data file paths, model configurations, and feature engineering modes.

Modules used:
- Pandera for schema validation
- Pydantic for configuration management and validation

"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandera.polars as pa
from pydantic import BaseModel, Field, FilePath
from pydantic.class_validators import root_validator


class DataValidationConfig(pa.DataFrameModel):
    """
    Validates the format of incoming data to ensure it conforms to the expected schema using pandera.

    Attributes:
        UserID (int): User ID.
        DeviceModel (str): Device model.
        OperatingSystem (str): Operating system used.
        AppUsageTime_min_day (int): Daily app usage time in minutes.
        ScreenOnTime_hours_day (float): Daily screen on time in hours.
        BatteryDrain_mAh_day (int): Daily battery drain in mAh.
        NumberOfAppsInstalled (int): Number of apps installed.
        DataUsage_MB_day (int): Daily data usage in MB.
        Age (int): Age in days.
        Gender (str): User's gender.
        UserBehaviorClass (int): Class label for user behavior.
    """

    UserID: int = pa.Field(description="User ID.")
    DeviceModel: str = pa.Field(description="Device Model.")
    OperatingSystem: str = pa.Field(description="Operating System.")
    AppUsageTime_min_day: int = pa.Field(description="App Usage Time in Minutes.")
    ScreenOnTime_hours_day: float = pa.Field(description="Screen On Time in Hours.")
    BatteryDrain_mAh_day: int = pa.Field(description="Battery Drain in Ah.")
    NumberOfAppsInstalled: int = pa.Field(description="Number of Apps installed.")
    DataUsage_MB_day: int = pa.Field(description="Data Usage in MB.")
    Age: int = pa.Field(description="Age in days.")
    Gender: str = pa.Field(description="Gender.")
    UserBehaviorClass: int = pa.Field(description="UserBehavior Class.")


class DataTransformationConfig(BaseModel):
    """
    Configuration for LazyTransformationPipeline class.

    Attributes:
        original_datapath (FilePath): Path to the original data file.
        mapping_name (str): Name of transformation mapping.
        intermediate_df_name (str): Name of intermediate data file.
        x_train_name (str): Name of training data file.
        y_train_name (str): Name of training data file.
        x_test_name (str): Name of testing data file.
        y_test_name (str): Name of testing data file.
        x_train_normalized_name (str): Name of training data file.
        x_test_normalized_name (str): Name of testing data file.
        x_train_standardized_name (str): Name of training data file.
        x_test_standardized_name (str): Name of testing data file.
        categorical_columns_to_transform (List[str]): Columns to transform from categorical to numeric.
        columns_to_drop (List[str]): Columns to drop from the DataFrame.
        mapping_file_path (Path): Path to the mapping file.
        ML_type (Literal[ "SVM","KNN","PCA","Gradient","Tree-Based","Naive"]): Model type to perform normalization just into numerical values or into categorical as well,
        normalize_df (bool): Whether to normalize the DataFrame.
        standardize_df (bool): Whether to standardize the DataFrame.
        number_iterations (int): Number of iterations performed by the cross validation model.
        cross_validation: (int): Number of cross validators.
        feature_engineering_dict (Dict[str, Union[float, int, str]]): Dictionary for feature engineering.
        tuned_parameters_path (Path): Path to the tuned parameters json file created after feature engineering is performed.
        transformed_intermediate_df_path (Path): Path to the transformed intermediate data file.
        transformed_train_df_path_x (Path): Path to save the transformed train_data X DataFrame.
        transformed_train_df_path_y (Path): Path to save the transformed train_data y DataFrame.
        transformed_test_df_path_x(Path): Path to save the transformed test_data X DataFrame.
        transformed_test_df_path_y (Path): Path to save the transformed test_data y DataFrame.
        target_column (str): Name of the target column.
        feature_mode (Literal['RandomSearch', "GridSearch", "BayesianOptim"]): The feature selection mode, which can be one of 'RandomSearch', 'GridSearch', or 'BayesianOptim'.
        transformed_normalized_df_path_train_x (Path): Path to save the transformed normalized train_data X DataFrame.
        transformed_standardized_df_path_train_x (Path): Path to save the transformed standardized train_data X DataFrame.
        transformed_normalized_df_path_test_x(Path): Path to save the transformed normalized test_data X DataFrame.
        transformed_standardized_df_path_test_x (Path): Path to save the transformed standardized test_data X DataFrame.

    """

    original_datapath: FilePath = Field(
        ..., description="Path to the original data folder"
    )

    mapping_name: str = Field(
        ..., description="Name of the mapping file categorical transformation."
    )
    intermediate_df_name: str = Field(
        ..., description="Name of transformed intermediate data file."
    )
    x_train_name: str = Field(
        ..., description="Name of the train_data data file after splitting."
    )
    y_train_name: str = Field(
        ..., description="Name of the train_data data file after splitting."
    )
    x_test_name: str = Field(
        ..., description="Name of the test_data data file after splitting."
    )
    y_test_name: str = Field(
        ..., description="Name of the test_data data file after splitting."
    )

    x_train_normalized_name: str = Field(
        ..., description="Name of the train_data data file after normalization."
    )
    x_test_normalized_name: str = Field(
        ..., description="Name of the test_data data file after normalization."
    )
    x_train_standardized_name: str = Field(
        ..., description="Name of the train_data data file after standardization."
    )
    x_test_standardized_name: str = Field(
        ..., description="Name of the test_data data file after standardization."
    )
    best_parameters_name: str = Field(
        ..., description="Name of tuned parameters json file created."
    )
    categorical_columns_to_transform: List[str] = Field(
        ...,
        description="List of columns to transform from categorical string to numerical",
    )
    columns_to_drop: List[str] = Field(..., description="List of columns to drop")
    mapping_file_path: Path = Field(
        ..., description="JSON file path for the categorical mapping."
    )

    ML_type: Literal["SVM", "KNN", "PCA", "Gradient", "Tree-Based", "Naive"] = Field(
        ...,
        description="Model type to perform normalization just into numerical values or into categorical as well",
    )
    normalize_df: bool = Field(..., description="Whether to normalize the data")

    standardize_df: bool = Field(..., description="Whether to standardize the data")

    apply_feature_selection: bool = Field(
        ..., description="If to apply or not the feature selection"
    )

    feature_selection: List[str] = Field(
        ..., description="The columns selected by correlation to y"
    )

    number_iterations: int = (
        Field(
            ...,
            description="Number of iterations performed by the cross validation model.",
        ),
    )

    cross_validation: int = Field(..., description="Number of cross validators")

    feature_engineering_dict: Dict[str, List[float | int | str] | float | int | str] = (
        Field(
            ...,
            description="Feature engineering dictionary specifying transformations for columns",
        )
    )

    tuned_parameters_path: Path = Field(
        ...,
        description="Path to the tuned parameters json file created after feature engineering is performed.",
    )

    transformed_intermediate_df_path: Path = Field(
        ..., description="Path to save the transformed intermediate DataFrame"
    )
    transformed_test_df_path_y: Path = Field(
        ..., description="Path to the transformed test_data data Y folder"
    )
    transformed_train_df_path_x: Path = Field(
        ..., description="Path to the transformed train_data data X folder"
    )
    transformed_train_df_path_y: Path = Field(
        ..., description="Path to the transformed train_data data y folder"
    )
    transformed_test_df_path_x: Path = Field(
        ..., description="Path to the transformed test_data data X folder"
    )
    transformed_normalized_df_path_train_x: Path = Field(
        ..., description="Path to save the transformed normalized train_data DataFrame"
    )
    transformed_standardized_df_path_train_x: Path = Field(
        ...,
        description="Path to save the transformed standardized train_data DataFrame",
    )
    transformed_normalized_df_path_test_x: Path = Field(
        ..., description="Path to save the transformed normalized test_data DataFrame"
    )
    transformed_standardized_df_path_test_x: Path = Field(
        ..., description="Path to save the transformed standardized test_data DataFrame"
    )
    target_column: str = Field(..., description="Name of the target column")
    feature_mode: Literal["RandomSearch", "GridSearch", "BayesianOptim"] = Field(
        ...,
        description="Feature mode selected from 'RandomSearch', 'GridSearch', or 'BayesianOptim'",
    )

    @root_validator(pre=True)
    def check_normalization_and_standardization(cls, values):
        """This function check that normalization and standardization are correctly implemented"""
        normalize_df = values["normalize_df"]
        standardized_df = values["standardize_df"]

        if normalize_df and standardized_df:
            raise ValueError("Both methods cannot be True at the same time.")
        return values


class TrainerConfig(BaseModel):
    """
    TrainerConfig is a Pydantic model that validates and manages the configuration for the training pipeline.

    Attributes:
        experiment_name (str): The name of the current experiment.
        x_train (FilePath): Path to the transformed training data (X) folder.
        x_test (FilePath): Path to the transformed testing data (X) folder.
        y_train (FilePath): Path to the training data (Y) folder.
        y_test (FilePath): Path to the transformed testing data (Y) folder.
        normalized_x_train (FilePath): Path to save the normalized training data DataFrame.
        standardized_x_train (FilePath): Path to save the standardized training data DataFrame.
        normalized_x_test (FilePath): Path to save the normalized testing data DataFrame.
        standardized_x_test (FilePath): Path to save the standardized testing data DataFrame.
        normalized_df (bool): Indicates whether to normalize the data.
        standardized_df (bool): Indicates whether to standardize the data.
        tuned_parameters (FilePath): File path to the JSON file containing the tuned parameters.
        model_path (Path): Path to save the trained machine learning model.
        metrics (Path): Path to the directory holding metrics for training and testing.

    Methods:
        check_normalization_and_standardization(cls, values):
            Validates that normalization and standardization are not enabled simultaneously.

    Raises:
        ValueError: If both `normalized_df` and `standardized_df` are set to `True`.
    """

    experiment_name: str = Field(..., description="The name of your current experiment")

    x_train: FilePath = Field(
        ..., description="Path to the transformed train_data data X folder"
    )
    x_test: FilePath = Field(
        ..., description="Path to the transformed test_data data X folder"
    )
    y_train: FilePath = Field(..., description="Path to the  train_data data Y folder")
    y_test: FilePath = Field(
        ..., description="Path to the transformed test_data data Y folder"
    )

    normalized_x_train: Optional[FilePath] = Field(
        None, description="Path to save the transformed normalized train_data DataFrame"
    )
    standardized_x_train: Optional[FilePath] = Field(
        None,
        description="Path to save the transformed standardized train_data DataFrame",
    )
    normalized_x_test: Optional[FilePath] = Field(
        None, description="Path to save the transformed normalized test_data DataFrame"
    )
    standardized_x_test: Optional[FilePath] = Field(
        None,
        description="Path to save the transformed standardized test_data DataFrame",
    )
    normalized_df: bool = Field(..., description="Whether to normalize the data")

    standardized_df: bool = Field(..., description="Whether to standardize the data")

    tuned_parameters: FilePath = Field(
        ..., description="File path to the json file with the tuned parameters"
    )

    model_path: Path = Field(..., description="The path to save the ML model trained.")

    average_mode: Literal["micro", "macro", "samples", "weighted", "binary"] = Field(
        ..., description="The average for classes in precision and recall"
    )

    metrics: Path = Field(
        ...,
        description="The path to the directory that holds the metrics for train and test.",
    )

    @root_validator(pre=True)
    def check_normalization_and_standardization(cls, values):
        """This function check that normalization and standardization are correctly implemented"""
        normalize_df = values["normalized_df"]
        standardized_df = values["standardized_df"]

        if normalize_df and standardized_df:
            raise ValueError("Both methods cannot be True at the same time.")
        return values
