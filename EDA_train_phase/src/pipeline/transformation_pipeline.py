"""
This module contains a class for data  transformation, and feature engineering pipelines. It is designed to handle the preprocessing of data for machine learning tasks, focusing on categorical transformation, normalization, standardization, and feature engineering. The module includes the following key components:

- LazyTransformationPipeline: A class that orchestrates the data transformation pipeline using Polars' lazy API. This class handles tasks such as categorical encoding, splitting data into train_data/test_data sets, applying normalization or standardization, and performing feature engineering.

The pipeline supports various machine learning models, including SVM, KNN, PCA, and tree-based models, and provides mechanisms for hyperparameter tuning using RandomSearch, GridSearch, or Bayesian Optimization.

Modules used:
- Polars for efficient data manipulation and transformation
- Scikit-learn for hyperparameter tuning
- Skopt for Bayesian optimization
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandera.polars as pa
import polars as pl
import polars.selectors as cs
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from torch.utils.hipify.hipify_python import mapping

from EDA_train_phase.src.hydra_config import load_hydra_config_from_root
from EDA_train_phase.src.logging_functions.logger import setup_logging
from EDA_train_phase.src.validation_classes.validation_configurations import (
    DataTransformationConfig,
    DataValidationConfig,
)

# TODO TESTING THE CLASSES WITHIN THE NOTEBOOK test if precommit works
# Changed the filepath to Path to take advantage of the train_config.yalm file
# Created Hyperparameters tuning methods
# Notebook Created
# DataValidationConfig has been updated to use Pandera for schema validation.
# Further integration with Polars and/or Pydantic is under consideration.


# CONSTANTS
LOG_FILE = Path("../logs/transformation_pipeline.log")
setup_logging(LOG_FILE)
CONFIG_PATH = "../conf"

CONFIG_ROOT = load_hydra_config_from_root(CONFIG_PATH)


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
        self.hydra_config: DictConfig = CONFIG_ROOT["transformation_config"]
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
            mapped = {value: idx for idx, value in enumerate(unique_values.to_series())}
            mappings[col] = mapped

        lf = (
            lf.with_columns(
                [
                    pl.col(col)
                    .replace(mappings[col])
                    .cast(pl.UInt8)
                    .alias(f"{col}_encoded")
                    for col in category_columns
                ]
            )
            .drop(category_columns + self.config.columns_to_drop)
            .sink_csv(
                self.config.transformed_intermediate_df_path
                / self.config.intermediate_df_name
            )
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
                f"Best hyperparameters saved at {self.config.best_model_params_path}/ "
                f"{self.config.best_parameters_name}{self.config.feature_mode}_{self.config.ML_type}.json."
            )

        else:
            with ValueError as e:
                logging.error(
                    f"You need to specify a model of classification. You can find the the accepted types in "
                    f"the DataTransformationConfig class"
                    f", in the attribute {self.config.ML_type} The error:\n{e}."
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
