"""
This module contains classes and functions for data validation, transformation, and feature engineering pipelines. It is designed to handle the preprocessing of data for machine learning tasks, focusing on categorical transformation, normalization, standardization, and feature engineering. The module includes the following key components:

- DataValidationConfig: A Pandera-based configuration class for validating incoming data and ensuring it adheres to an expected schema.
- DataTransformationConfig: A Pydantic-based configuration class for managing transformation settings, including data file paths, model configurations, and feature engineering modes.
- LazyTransformationPipeline: A class that orchestrates the data transformation pipeline using Polars' lazy API. This class handles tasks such as categorical encoding, splitting data into train/test sets, applying normalization or standardization, and performing feature engineering.

The pipeline supports various machine learning models, including SVM, KNN, PCA, and tree-based models, and provides mechanisms for hyperparameter tuning using RandomSearch, GridSearch, or Bayesian Optimization.

Modules used:
- Pandera for schema validation
- Polars for efficient data manipulation and transformation
- Pydantic for configuration management and validation
- Scikit-learn for hyperparameter tuning
- Skopt for Bayesian optimization
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, FilePath
from pydantic.class_validators import root_validator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from logger import setup_logging

# TODO TESTING THE CLASSES WITHIN THE NOTEBOOK
# Changed the filepath to Path to take advantage of the trainconfig.yalm file
# Created Hyperparameters tuning methods
# Notebook Created
# DataValidationConfig has been updated to use Pandera for schema validation.
# Further integration with Polars and/or Pydantic is under consideration.


# CONSTANTS
LOG_FILE = Path("./logs/application.log")
setup_logging(LOG_FILE)
initialize(config_path="./conf/transformation_config/")
HYDRA_CONFIG = compose(config_name="transformation_config")
CONFIG_DICT = OmegaConf.to_object(HYDRA_CONFIG)


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
        x_train_normalized (str): Name of training data file.
        x_test_normalized (str): Name of testing data file.
        x_train_standardized (str): Name of training data file.
        x_test_standardized (str): Name of testing data file.
        categorical_columns_to_transform (List[str]): Columns to transform from categorical to numeric.
        columns_to_drop (List[str]): Columns to drop from the DataFrame.
        mapping_file_path (Path): Path to the mapping file.
        ML_type (Literal[ "SVM","KNN","PCA","Gradient","Tree-Based","Naive"]): Model type to perform normalization just into numerical values or into categorical as well,
        normalize_df (bool): Whether to normalize the DataFrame.
        standardized_df (bool): Whether to standardize the DataFrame.
        feature_engineering_dict (Dict[str, Union[float, int, str]]): Dictionary for feature engineering.
        tuned_parameters (FilePath): Path to the tuned parameters json file created after feature engineering is performed.
        transformed_intermediate_df_path (Path): Path to the transformed intermediate data file.
        transformed_train_df_path_x (Path): Path to save the transformed train X DataFrame.
        transformed_train_df_path_y (Path): Path to save the transformed train y DataFrame.
        transformed_test_df_path_x(Path): Path to save the transformed test X DataFrame.
        transformed_test_df_path_y (Path): Path to save the transformed test y DataFrame.
        target_column (str): Name of the target column.
        feature_mode (Literal['RandomSearch', "GridSearch", "BayesianOptim"]): The feature selection mode, which can be one of 'RandomSearch', 'GridSearch', or 'BayesianOptim'.
        transformed_normalized_df_path_train_x (Path): Path to save the transformed normalized train X DataFrame.
        transformed_standardized_df_path_train_x (Path): Path to save the transformed standardized train X DataFrame.
        transformed_normalized_df_path_test_x(Path): Path to save the transformed normalized test X DataFrame.
        transformed_standardized_df_path_test_x (Path): Path to save the transformed standardized test X DataFrame.



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
        ..., description="Name of the train data file after splitting."
    )
    y_train_name: str = Field(
        ..., description="Name of the train data file after splitting."
    )
    x_test_name: str = Field(
        ..., description="Name of the test data file after splitting."
    )
    y_test_name: str = Field(
        ..., description="Name of the test data file after splitting."
    )

    x_train_normalized: str = Field(
        ..., description="Name of the train data file after normalization."
    )
    x_test_normalized: str = Field(
        ..., description="Name of the test data file after normalization."
    )
    x_train_standardized: str = Field(
        ..., description="Name of the train data file after standardization."
    )
    x_test_standardized: str = Field(
        ..., description="Name of the test data file after standardization."
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

    standardized_df: bool = Field(..., description="Whether to standardize the data")

    feature_engineering_dict: Dict[
        str, List[float | int | str] | float | int | str
    ] = Field(
        ...,
        description="Feature engineering dictionary specifying transformations for columns",
    )

    tuned_parameters: Path = Field(
        ...,
        description="Path to the tuned parameters json file created after feature engineering is performed.",
    )

    transformed_intermediate_df_path: Path = Field(
        ..., description="Path to save the transformed intermediate DataFrame"
    )
    transformed_test_df_path_y: Path = Field(
        ..., description="Path to the transformed test data Y folder"
    )
    transformed_train_df_path_x: Path = Field(
        ..., description="Path to the transformed train data X folder"
    )
    transformed_train_df_path_y: Path = Field(
        ..., description="Path to the transformed train data y folder"
    )
    transformed_test_df_path_x: Path = Field(
        ..., description="Path to the transformed test data X folder"
    )
    transformed_normalized_df_path_train_x: Path = Field(
        ..., description="Path to save the transformed normalized train DataFrame"
    )
    transformed_standardized_df_path_train_x: Path = Field(
        ..., description="Path to save the transformed standardized train DataFrame"
    )
    transformed_normalized_df_path_test_x: Path = Field(
        ..., description="Path to save the transformed normalized test DataFrame"
    )
    transformed_standardized_df_path_test_x: Path = Field(
        ..., description="Path to save the transformed standardized test DataFrame"
    )
    target_column: str = Field(..., description="Name of the target column")
    feature_mode: Literal["RandomSearch", "GridSearch", "BayesianOptim"] = Field(
        ...,
        description="Feature mode selected from 'RandomSearch', 'GridSearch', or 'BayesianOptim'",
    )

    @root_validator(pre=True)
    def check_normalization_and_standardization(cls, values):
        normalize_df = values["normalize_df"]
        standardized_df = values["standardized_df"]

        if normalize_df and standardized_df:
            raise ValueError("Both methods cannot be True at the same time.")
        return values


class LazyTransformationPipeline:
    """
    A pipeline class to perform efficient data transformations on dataframes using Polars' lazy API.
    The class reads raw data, validates it, applies categorical to numerical encoding, splits into
    training and testing datasets, and optionally normalizes or standardizes the data. It also
    supports feature engineering using Random Search, Grid Search, and Bayesian Optimization.

    Attributes:
        config: An instance of the DataTransformationConfig class, holding configuration settings.
        df_validation: An instance of the DataValidationConfig class, used to validate the data.
        model: The machine learning model used for feature engineering.
    """

    def __init__(self):
        """
        Initializes the LazyTransformationPipeline with default configurations for data transformation,
        validation, and feature engineering. It attempts to validate the dataframe schema based on the
        provided configuration. If validation fails, an error message is printed.

        Raises:
            SchemaError: If the dataframe schema does not match the expected schema defined in the configuration.
        """
        self.df_validation: DataValidationConfig = DataValidationConfig
        self.hydra_config: DictConfig = HYDRA_CONFIG
        self.model = None
        self.search_class = None
        try:
            self.config: DataTransformationConfig = DataTransformationConfig(
                **self.hydra_config
            )

            logging.info("Valid transformation configuration found.")
        except ValueError as e:
            logging.error(f"Failed transformation configuration yalm file with {e}")
            raise e
        try:
            pl.scan_csv(self.config.original_datapath).pipe(self.df_validation.validate)

            logging.info("DataFrame Validation is correct")
        except pa.errors.SchemaError as e:
            logging.error(f"Dataframe validation failed with {e}")
            raise e

    def df_categorical_to_numerical(self) -> None:
        """
        Transforms categorical columns into numeric representations using consistent mappings. The transformed
        dataframe is saved to the specified path in the configuration.

        The method:
            - Loads the data lazily using Polars
            - Creates mappings for categorical columns to numerical values
            - Drops original categorical columns and any columns specified in `columns_to_drop`
            - Saves the transformed dataframe and the mappings to specified paths

        Returns:
            None
        """
        category_columns = self.config.categorical_columns_to_transform

        lf = pl.scan_csv(self.config.original_datapath)

        mappings = {}
        for col in category_columns:
            unique_values = lf.select(pl.col(col).unique()).collect()
            mapping = {
                value: idx for idx, value in enumerate(unique_values.to_series())
            }
            mappings[col] = mapping
            lf = lf.with_columns(
                pl.col(col).map_elements(mapping).alias(f"{col}_encoded")
            )

        lf = lf.drop(category_columns + self.config.columns_to_drop)

        # Save the transformed LazyFrame
        lf.sink_csv(
            self.config.transformed_intermediate_df_path
            / self.config.intermediate_df_name
        )

        with open(self.config.mapping_file_path / self.config.mapping_name, "w") as f:
            json.dump(mappings, f)

        logging.info(
            f"Categorical columns transformed and saved at {self.config.transformed_intermediate_df_path / self.config.intermediate_df_name}."
        )

    def split_train_test(
        self, random_state: int = 42, train_fraction: float = 0.75
    ) -> None:
        """
        Splits the transformed dataset into training and testing sets based on the specified fraction.

        The method:
            - Loads the transformed dataframe lazily
            - Adds a row number for shuffling and splitting
            - Splits the data based on the row number into training and testing sets
            - Saves the resulting datasets (X_train, X_test, y_train, y_test) to the specified paths

        Args:
            random_state (int, optional): The seed for random operations. Defaults to 42.
            train_fraction (float, optional): The fraction of data to use for training. Defaults to 0.75.

        Returns:
            None
        """
        if not (0 <= train_fraction <= 1):
            raise ValueError("train_fraction must be between 0 and 1.")

        lazy_df = (
            pl.scan_csv(self.config.transformed_intermediate_df_path)
            .with_columns(pl.arange(0, pl.count()).alias("row_nr"))
            .with_columns(pl.all().shuffle(seed=random_state))
        )

        train_lazy_df = lazy_df.filter(
            pl.col("row_nr") < pl.col("row_nr").max() * train_fraction
        )
        test_lazy_df = lazy_df.filter(
            pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction
        )

        x_train = train_lazy_df.drop([self.config.target_column, "row_nr"])
        x_test = test_lazy_df.drop([self.config.target_column, "row_nr"])
        y_train = train_lazy_df.select(self.config.target_column)
        y_test = test_lazy_df.select(self.config.target_column)

        x_train.sink_csv(
            self.config.transformed_train_df_path_x / self.config.x_train_name,
            index=False,
        )
        x_test.sink_csv(
            self.config.transformed_test_df_path_x / self.config.x_test_name,
            index=False,
        )
        y_train.sink_csv(
            self.config.transformed_train_df_path_y / self.config.y_train_name,
            index=False,
        )
        y_test.sink_csv(
            self.config.transformed_test_df_path_y / self.config.y_test_name,
            index=False,
        )

    def prepare_for_normalize_or_standardize(
        self,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Prepares the data to be used in normalization or standardization."""
        # Read data lazily
        lazy_df_train = pl.scan_csv(
            self.config.transformed_train_df_path_x / self.config.x_train_name
        )
        lazy_df_test = pl.scan_csv(
            self.config.transformed_test_df_path_x / self.config.x_test_name
        )

        # Create variables
        train_categorical_columns = lazy_df_train.select(
            cs.contains("_encoded")
        ).columns
        train_continuous_columns = lazy_df_train.select(
            cs.exclude(cs.contains("_encoded"))
        ).columns
        test_categorical_columns = lazy_df_test.select(cs.contains("_encoded")).columns
        test_continuous_columns = lazy_df_test.select(
            cs.exclude(cs.contains("_encoded"))
        ).columns
        train_all_columns = train_categorical_columns + train_continuous_columns
        test_all_columns = test_categorical_columns + test_continuous_columns

        return (
            train_continuous_columns,
            test_continuous_columns,
            train_all_columns,
            test_all_columns,
        )

    def normalize(self) -> None:
        """
        Performs column-wise normalization [0, 1] for continuous columns only.
        If the model is SVM, KNN, or PCA, normalizes categorical encoded columns as well.
        Otherwise, skips normalization for categorical columns.

        Args:
        model_type (str): The model type used for prediction (e.g., "SVM", "KNN", "PCA").
        """
        # Specify models that will normalize the categorical columns already converted
        normalize_categorical = ["SVM", "KNN", "PCA"]

        # Read data lazily
        lazy_df_train = pl.scan_csv(
            self.config.transformed_train_df_path_x / self.config.x_train_name
        )
        lazy_df_test = pl.scan_csv(
            self.config.transformed_test_df_path_x / self.config.x_test_name
        )

        (
            train_continuous_columns,
            test_continuous_columns,
            train_all_columns,
            test_all_columns,
        ) = self.prepare_for_normalize_or_standardize()
        # Check if the model is distance, gradient, or scale-sensitive
        if self.config.model_type in normalize_categorical:
            logging.info(
                f"Model type '{self.config.model_type}' detected. Normalizing categorical columns as well."
            )

            # Normalize categorical encoded columns (if using SVM, KNN, or PCA)
            train_normalized = lazy_df_train.select(
                [
                    (pl.col(col) - pl.col(col).min())
                    / (pl.col(col).max() - pl.col(col).min())
                    for col in train_all_columns
                ]
            )
            test_normalized = lazy_df_test.select(
                [
                    (pl.col(col) - pl.col(col).min())
                    / (pl.col(col).max() - pl.col(col).min())
                    for col in test_all_columns
                ]
            )
        else:
            logging.info(
                f"Model type '{self.config.model_type}' detected. Skipping normalization for categorical columns."
            )
            train_normalized = lazy_df_train.select(
                [
                    (pl.col(col) - pl.col(col).min())
                    / (pl.col(col).max() - pl.col(col).min())
                    for col in train_continuous_columns
                ]
                + [
                    pl.col(c)
                    for c in lazy_df_train.columns
                    if c not in train_continuous_columns
                ]
            )
            test_normalized = lazy_df_test.select(
                [
                    (pl.col(col) - pl.col(col).min())
                    / (pl.col(col).max() - pl.col(col).min())
                    for col in test_continuous_columns
                ]
                + [
                    pl.col(c)
                    for c in lazy_df_test.columns
                    if c not in test_continuous_columns
                ]
            )

        # Save the normalized dataset
        train_normalized.sink_csv(
            self.config.transformed_normalized_df_path_train_x
            / self.config.x_train_normalized
        )
        test_normalized.sink_csv(
            self.config.transformed_normalized_df_path_test_x
            / self.config.x_test_normalized
        )

    def standardize(self) -> None:
        """
        Performs column-wise standardization [0, 1] for continuous columns only.
        If the model is SVM, KNN, or PCA, normalizes categorical encoded columns as well.
        Otherwise, skips normalization for categorical columns.

        Args:
        model_type (str): The model type used for prediction (e.g., "SVM", "KNN", "PCA").
        """
        # Specify models that will standardize the categorical columns already converted
        standardize_categorical: list[str] = ["SVM", "KNN", "PCA"]

        # Read data lazily
        lazy_df_train = pl.scan_csv(
            self.config.transformed_train_df_path_x / self.config.x_train_name
        )
        lazy_df_test = pl.scan_csv(
            self.config.transformed_test_df_path_x / self.config.x_test_name
        )
        (
            train_continuous_columns,
            test_continuous_columns,
            train_all_columns,
            test_all_columns,
        ) = self.prepare_for_normalize_or_standardize()

        # Check if the model is distance, gradient, or scale-sensitive
        if self.config.model_type in standardize_categorical:
            logging.info(
                f"Model type '{self.config.model_type}' detected. Normalizing categorical columns as well."
            )

            # Normalize categorical encoded columns (if using SVM, KNN, or PCA)
            train_standardized = lazy_df_train.select(
                [
                    (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
                    for col in train_all_columns
                ]
            )
            test_standardized = lazy_df_test.select(
                [
                    (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
                    for col in test_all_columns
                ]
            )
        else:
            logging.info(
                f"Model type '{self.config.model_type}' detected. Skipping normalization for categorical columns."
            )
            train_standardized = lazy_df_train.select(
                [
                    (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
                    for col in train_continuous_columns
                ]
                + [
                    pl.col(c)
                    for c in lazy_df_train.columns
                    if c not in train_continuous_columns
                ]
            )
            test_standardized = lazy_df_test.select(
                [
                    (pl.col(col) - pl.col(col).min()) / pl.col(col).std()
                    for col in test_continuous_columns
                ]
                + [
                    pl.col(c)
                    for c in lazy_df_test.columns
                    if c not in test_continuous_columns
                ]
            )

            # Save the normalized dataset
        train_standardized.sink_csv(
            self.config.transformed_standardized_df_path_train_x
            / self.config.x_train_standardized
        )
        test_standardized.sink_csv(
            self.config.transformed_standardized_df_path_test_x
            / self.config.x_test_standardized
        )

    def _apply_feature_search(self, search_class) -> None:
        """
        General function to apply any feature search method (RandomizedSearch, GridSearch, or BayesSearch).

        Args:
            search_class (class): The search class to use (RandomizedSearchCV, GridSearchCV, or BayesSearchCV).

        Raises:
            ValueError: If no model is provided.
        """
        if self.model is not None and self.search_class is not None:
            clf = search_class(self.model, self.config.tuned_parameters)
            if self.config.standarized_df:
                search = clf.fit(
                    self.config.transformed_standarized_df_path_train_X,
                    self.config.transformed_train_df_path_y,
                )
            elif self.config.normalize:
                search = clf.fit(
                    self.config.transformed_normalized_df_path_test_X,
                    self.config.transformed_train_df_path_y,
                )

            else:
                search = clf.fit(
                    self.config.transformed_train_df_path_x,
                    self.config.transformed_train_df_path_y,
                )
            best_params = search.best_params_
            with open(
                self.config.tunable_parameters_path
                / f"{self.config.best_parameters_name}{self.config.feature_mode}_{self.config.ML_type}.json",
                "w",
            ) as f:
                json.dump(best_params, f)
            logging.info(
                f"Best hyperparameters saved at {self.config.best_model_params_path}/ {self.config.best_parameters_name}{self.config.feature_mode}_{self.config.ML_type}.json."
            )

        else:
            with ValueError as e:
                logging.error(
                    f"You need to specify a model of classification. You can find the the accepted types in the DataTransformationConfig class, in the attribute {self.config.ML_type} The error:\n{e}."
                )
                raise ValueError(e)

    def apply_feature_engineering(self) -> None:
        """
        Applies the specified feature engineering method.
        """
        mode = self.config.feature_mode
        if mode == "RandomSearch":
            self._apply_feature_search(RandomizedSearchCV)
        elif mode == "GridSearch":
            self._apply_feature_search(GridSearchCV)
        else:
            self._apply_feature_search(BayesSearchCV)

    def run(self) -> None:
        """
        Executes the full pipeline, including categorical transformation, splitting,
        normalization/standardization, and feature engineering.
        """
        logging.info("Pipeline running...")
        try:
            self.df_categorical_to_numerical()
            logging.info("Categorical Data Transformed and saved")

            self.split_train_test()
            logging.info("Split Data Transformed and saved")

            # Directly call normalization and standardization methods

            if self.config.normalize_df:
                self.normalize()
                logging.info("Normalization applied.")

            if self.config.standarized_df:
                self.standardize()
                logging.info("Standardization applied.")
            else:
                logging.info("Skipped Normalization / Standardization.")

            self.apply_feature_engineering()
            logging.info("Feature engineering applied.")

            logging.info("Pipeline Finished.")
        except Exception as e:
            logging.error(f"Pipeline failure at {e}")
            raise e
