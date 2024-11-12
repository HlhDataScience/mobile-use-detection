import json
import logging
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, FilePath, ValidationError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)
# Class Validations for Data input and data transformation


class DataValidationConfig(BaseModel):
    """
    Validates the format of incoming data to ensure it conforms to the expected schema.

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

    UserID: int = Field(..., description="User ID.")
    DeviceModel: str = Field(..., description="Device Model.")
    OperatingSystem: str = Field(..., description="Operating System.")
    AppUsageTime_min_day: int = Field(..., description="App Usage Time in Minutes.")
    ScreenOnTime_hours_day: float = Field(..., description="Screen On Time in Hours.")
    BatteryDrain_mAh_day: int = Field(..., description="Battery Drain in Ah.")
    NumberOfAppsInstalled: int = Field(..., description="Number of Apps installed.")
    DataUsage_MB_day: int = Field(..., description="Data Usage in MB.")
    Age: int = Field(..., description="Age in days.")
    Gender: str = Field(..., description="Gender.")
    UserBehaviorClass: int = Field(..., description="UserBehavior Class.")

    @classmethod
    def validate_data(cls, data: Dict[str, Any]) -> bool:
        """
        Validates a dictionary of data to ensure it matches the schema.

        Args:
            data (Dict[str, Any]): Data to validate.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        try:
            cls(**data)
            return True
        except ValidationError as e:
            logger.error(f"Validation error: {e}")

    @classmethod
    def validate_csv(
        cls,
        filepath: FilePath,
        stop_on_invalid_row: bool = True,
        raise_on_invalid: bool = True,
    ) -> bool:
        """
        Validates each row of a CSV file against the schema.

        Args:
            filepath (Path): Path to the CSV file.
            stop_on_invalid_row (bool): If True, will stop processing after the first invalid row.
            raise_on_invalid (bool): If True, will raise an exception after finding an invalid row.

        Returns:
            bool: True if all rows are valid, False otherwise.

        Raises:
            ValidationErrorException: If raise_on_invalid is True and an invalid row is found.
        """
        all_valid = True

        # Use 'with' to automatically handle file opening and closing
        df = pl.read_csv(filepath)
        dfdict = df.to_dicts()
        for row in dfdict:
            if not cls.validate_data(row):
                all_valid = False
                logger.error(f"Invalid row: {row}")
                if raise_on_invalid:
                    raise TypeError(f"Invalid row found: {row}")
                if stop_on_invalid_row:
                    return False

        return all_valid

    @classmethod
    def validate_and_serialize(
        cls,
        filepath: FilePath,
        json_schema_filepath: str,
        json_schema_name: str = "json_schema.json",
        serialized_data: str = "validated_serialized_data.json",
    ) -> str:
        """
        Validates each row of a CSV file against the schema and serialize the file to be ready to use

        Args:
            filepath (Path): Path to the CSV file.
            json_schema_filepath (str): Path to the JSON Schema file.
            json_schema_name (str): Name of the JSON Schema file.
            serialized_data (str): Path to the serialized JSON Schema file.

        Returns:
            bool: True if all rows are valid, False otherwise.
            Dict[str, Any]: Dictionary of serialized data.

        Raises:
            ValidationErrorException: If raise_on_invalid is True and an invalid row is found.
        """

        all_valid = True
        schema_path: FilePath = str(
            os.path.join(json_schema_filepath, json_schema_name)
        )
        serialized_data_path: FilePath = str(
            os.path.join(json_schema_filepath, serialized_data)
        )

        df = pl.read_csv(filepath)
        df_dict = df.to_dicts()
        for row in df_dict:
            if not cls.validate_data(row):
                all_valid = False
                logger.error(f"Invalid row: {row}")

                raise TypeError(f"Invalid row found: {row}")

        with open(schema_path, "w") as f:
            schema_to_json = cls.model_json_schema()
            json.dump(schema_to_json, f)

        with open(serialized_data_path, "w") as f:
            json.dump(df_dict, f)

        return print(
            f"All the files are valid : {all_valid}\n\nJSON schema file saved at {schema_path}\n\n JSON serialized data at {schema_path}"
        )


class DataTransformationConfig(BaseModel):
    """
    Configuration for data transformation steps.

    Attributes:
        original_datapath (FilePath): Path to the original data file.
        categorical_columns_to_transform (List[str]): Columns to transform from categorical to numeric.
        columns_to_drop (List[str]): Columns to drop from the DataFrame.
        normalize_df (bool): Whether to normalize the DataFrame.
        feature_engineering_dict (Dict[str, Union[float, int, str]]): Dictionary for feature engineering.
        transformed_train_df_path (FilePath): Path to save the transformed train DataFrame.
        transformed_test_df_path (FilePath): Path to save the transformed test DataFrame.
        target_column (str): Name of the target column.
    """

    original_datapath: FilePath = Field(
        ..., description="Path to the original data folder"
    )
    categorical_columns_to_transform: List[str] = Field(
        ...,
        description="List of columns to transform from categorical string to numerical",
    )
    columns_to_drop: List[str] = Field(..., description="List of columns to drop")
    normalize_df: bool = Field(..., description="Whether to normalize the data")
    feature_engineering_dict: Dict[str, float | int | str] = Field(
        ..., description="Feature engineering dict"
    )
    transformed_train_df_path: FilePath = Field(
        ..., description="Path to the transformed train data folder"
    )
    transformed_test_df_path: FilePath = Field(
        ..., description="Path to the transformed test data folder"
    )
    target_column: str = Field(..., description="target column name")


# Transformation functions:


def split(
    config: DataTransformationConfig, return_files: bool, random_state: str = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the data into train and test sets."""
    print("Loading data...")
    df = pl.read_csv(config.original_datapath)
    y = df[config.target_column]
    x = df.drop(config.target_column)

    train_df, test_df = train_test_split(x, y, random_state=random_state, stratify=y)

    print("Saving split data...")
    train_df.to_csv(config.transformed_train_df_path, index=False)
    test_df.to_csv(config.transformed_test_df_path, index=False)
    if return_files:
        return train_df, test_df


def df_categorical_to_numerical(
    df: pl.DataFrame, config: DataTransformationConfig, return_corr_matrix: bool
) -> Union[Tuple[pl.DataFrame, List[str], np.ndarray], Tuple[pl.DataFrame, List[str]]]:
    """
    Encodes categorical columns as numeric values and optionally returns a correlation matrix.

    Args:
        df (pl.DataFrame): The DataFrame containing categorical columns.
        config (DataTransformationConfig): Configuration with columns to transform and drop.
        return_corr_matrix (bool): If True, returns a correlation matrix.

    Returns:
        Union[Tuple[pl.DataFrame, List[str], np.ndarray], Tuple[pl.DataFrame, List[str]]]:
        - If return_corr_matrix is True, returns a tuple with the DataFrame, column names, and correlation matrix.
        - If return_corr_matrix is False, returns a tuple with the DataFrame and column names.
    """
    category_columns = config.categorical_columns_to_transform

    # Cast categorical columns to integer codes and create new columns with the encoded values
    df = df.clone()
    df = df.with_columns(
        [
            pl.col(col).cast(pl.Categorical).to_physical().alias(f"{col}_encoded")
            for col in category_columns
        ]
    )
    df = df.drop(config.categorical_columns_to_transform)
    df = df.drop(config.columns_to_drop)

    if return_corr_matrix:
        correlation_matrix = np.corrcoef(df.to_numpy(), rowvar=False)
        return df, df.columns, correlation_matrix

    return df, df.columns
