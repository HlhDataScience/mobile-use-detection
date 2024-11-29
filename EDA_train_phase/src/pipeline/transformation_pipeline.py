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
from typing import List, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV

from EDA_train_phase.src.abstractions.ABC_Pipeline import BasicPipeline
from EDA_train_phase.src.validation_classes.validation_interfaces import (
    HydraConfLoader,
    PanderaValidationModel,
    PydanticConfigModel,
)


class LazyTransformationPipeline(BasicPipeline):
    """
    A pipeline class to perform efficient data transformations on dataframes using Polars' lazy API.
    The class reads raw data, validates it, applies categorical to numerical encoding, splits into
    training and testing datasets, and optionally normalizes or standardizes the data. It also
    supports feature engineering using Random Search, Grid Search, and Bayesian Optimization.

    Attributes:
        validation_model (DataValidationConfig): An instance of the DataValidationConfig class, used to validate the data.
        config_model (DataTransformationConfig): An instance of the DataTransformationConfig class, holding configuration settings.
        config_data (DictConfig): The omegaconf DictObject that holds the parameters of the yalm configuration.
        apply_custom_function (bool): a flag to use a custom function to validate the dataframe. Default = False

    """

    def __init__(
        self,
        validation_model: PanderaValidationModel,
        config_model: PydanticConfigModel,
        config_loader: HydraConfLoader,
        config_name: str,
        config_section: str,
        apply_custom_function: bool,
        model: BaseEstimator,
    ):
        """
        Initializes the LazyTransformationPipeline subclass with default super configurations for data transformation,
        validation, and feature engineering. It attempts to validate the dataframe schema based on the
        provided configuration. If validation fails, an error message is printed.

        Raises:
            SchemaError: If the dataframe schema does not match the expected schema defined in the configuration.
        """
        super().__init__(
            validation_model=validation_model,
            config_model=config_model,
            config_loader=config_loader,
            config_name=config_name,
            config_section=config_section,
            apply_custom_function=apply_custom_function,
        )
        self.model = model
        self.search_class = self.valid_config.feature_mode

    def categorical_encoding(self) -> None:
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
        category_columns = self.valid_config.categorical_columns_to_transform

        lf = pl.scan_csv(self.valid_config.original_datapath)

        mappings = {}
        for col in category_columns:
            unique_values = lf.select(pl.col(col).unique()).collect()
            mapped = {value: idx for idx, value in enumerate(unique_values.to_series())}
            mappings[col] = mapped

        (
            lf.with_columns(
                [
                    pl.col(col)
                    .replace(mappings[col])
                    .cast(pl.Int32)
                    .alias(f"{col}_encoded")
                    for col in category_columns
                ]
            )
            .drop(category_columns + self.valid_config.columns_to_drop)
            .collect()
            .write_csv(
                self.valid_config.transformed_intermediate_df_path
                / self.valid_config.intermediate_df_name
            )
        )

        with open(
            self.valid_config.mapping_file_path / self.valid_config.mapping_name,
            mode="w",
        ) as f:
            json.dump(mappings, f)

        logging.info(
            f"Categorical columns transformed and saved at {self.valid_config.transformed_intermediate_df_path / self.valid_config.intermediate_df_name}."
        )

    def _feature_selection(
        self, train: pl.LazyFrame, test: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Selects the columns specified to be part of the train and test split."""

        train = train.select(
            [pl.col(col) for col in self.valid_config.feature_selection]
        )
        test = test.select([pl.col(col) for col in self.valid_config.feature_selection])
        return train, test

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
            pl.scan_csv(
                self.valid_config.transformed_intermediate_df_path
                / self.valid_config.intermediate_df_name
            )
            .with_columns(pl.arange(0, pl.count()).alias("row_nr"))
            .with_columns(pl.all().shuffle(seed=random_state))
        )

        train_lazy_df = lazy_df.filter(
            pl.col("row_nr") < pl.col("row_nr").max() * train_fraction
        )
        test_lazy_df = lazy_df.filter(
            pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction
        )
        x_train = train_lazy_df.drop([self.valid_config.target_column, "row_nr"])
        x_test = test_lazy_df.drop([self.valid_config.target_column, "row_nr"])
        y_train = train_lazy_df.select(self.valid_config.target_column)
        y_test = test_lazy_df.select(self.valid_config.target_column)
        if self.valid_config.apply_feature_selection:
            x_train, x_test = self._feature_selection(x_train, x_test)
        x_train.collect().write_csv(
            self.valid_config.transformed_train_df_path_x
            / self.valid_config.x_train_name,
        )
        x_test.collect().write_csv(
            self.valid_config.transformed_test_df_path_x
            / self.valid_config.x_test_name,
        )
        y_train.collect().write_csv(
            self.valid_config.transformed_train_df_path_y
            / self.valid_config.y_train_name,
        )
        y_test.collect().write_csv(
            self.valid_config.transformed_test_df_path_y
            / self.valid_config.y_test_name,
        )

    def custom_validate(self):
        pass

    def scaling(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Prepares the data to be used in normalization or standardization."""
        # Read data lazily
        lazy_df_train = pl.scan_csv(
            self.valid_config.transformed_train_df_path_x
            / self.valid_config.x_train_name
        )
        lazy_df_test = pl.scan_csv(
            self.valid_config.transformed_test_df_path_x / self.valid_config.x_test_name
        )
        # Create variables
        train_categorical_columns = (
            lazy_df_train.select(cs.contains("_encoded")).collect_schema().names()
        )
        train_continuous_columns = (
            lazy_df_train.select(cs.exclude(cs.contains("_encoded")))
            .collect_schema()
            .names()
        )
        test_categorical_columns = (
            lazy_df_test.select(cs.contains("_encoded")).collect_schema().names()
        )
        test_continuous_columns = (
            lazy_df_test.select(cs.exclude(cs.contains("_encoded")))
            .collect_schema()
            .names()
        )
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
        # Raise error if the method is not specified in the transformation config.
        if not self.valid_config.normalize_df:
            logging.error(
                f"The value of normalized_df is {self.valid_config.normalize_df} You need to set True to use it"
            )
            raise ValueError
        else:
            # Read data lazily
            lazy_df_train = pl.scan_csv(
                self.valid_config.transformed_train_df_path_x
                / self.valid_config.x_train_name
            )
            lazy_df_test = pl.scan_csv(
                self.valid_config.transformed_test_df_path_x
                / self.valid_config.x_test_name
            )
            (
                train_continuous_columns,
                test_continuous_columns,
                train_all_columns,
                test_all_columns,
            ) = self.scaling()
            # Check if the model is distance, gradient, or scale-sensitive
            if self.valid_config.ML_type in normalize_categorical:
                logging.info(
                    f"Model type '{self.valid_config.ML_type}' detected. Normalizing categorical columns as well."
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
                    f"Model type '{self.valid_config.ML_type}' detected. Skipping normalization for categorical columns."
                )
                train_normalized = lazy_df_train.select(
                    [
                        (pl.col(col) - pl.col(col).min())
                        / (pl.col(col).max() - pl.col(col).min())
                        for col in train_continuous_columns
                    ]
                    + [
                        pl.col(c).cast(pl.Float32)
                        for c in lazy_df_train.columns
                        if c not in train_continuous_columns
                    ]
                )
                test_normalized = lazy_df_test.select(
                    [
                        (pl.col(col) - pl.col(col).min())
                        / (pl.col(col).max() - pl.col(col).min())
                        for col in train_continuous_columns
                    ]
                    + [
                        pl.col(c)
                        for c in lazy_df_test.columns
                        if c not in test_continuous_columns
                    ]
                )
            # Save the normalized dataset
            train_normalized.collect().write_csv(
                self.valid_config.transformed_normalized_df_path_train_x
                / self.valid_config.x_train_normalized_name
            )
            test_normalized.collect().write_csv(
                self.valid_config.transformed_normalized_df_path_test_x
                / self.valid_config.x_test_normalized_name
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

        # Raise error if the method is not specified in the transformation config.
        if not self.valid_config.standardized_df:
            logging.error(
                f"The value of standardized_df is {self.valid_config.standardized_df} You need to set True to use it"
            )
            raise ValueError
        else:
            # Read data lazily
            lazy_df_train = pl.scan_csv(
                self.valid_config.transformed_train_df_path_x
                / self.valid_config.x_train_name
            )
            lazy_df_test = pl.scan_csv(
                self.valid_config.transformed_test_df_path_x
                / self.valid_config.x_test_name
            )
            (
                train_continuous_columns,
                test_continuous_columns,
                train_all_columns,
                test_all_columns,
            ) = self.scaling()
            # Check if the model is distance, gradient, or scale-sensitive
            if self.valid_config.ML_type in standardize_categorical:
                logging.info(
                    f"Model type '{self.valid_config.ML_type}' detected. Normalizing categorical columns as well."
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
                    f"Model type '{self.valid_config.ML_type}' detected. Skipping normalization for categorical columns."
                )
                train_standardized = lazy_df_train.select(
                    [
                        (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
                        for col in train_continuous_columns
                    ]
                    + [
                        pl.col(c).cast(pl.Float32)
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
            train_standardized.collect().write_csv(
                self.valid_config.transformed_standardized_df_path_train_x
                / self.valid_config.x_train_standardized_name
            )
            test_standardized.collect().write_csv(
                self.valid_config.transformed_standardized_df_path_test_x
                / self.valid_config.x_test_standardized_name
            )

    def _apply_feature_search(self, search_class=None) -> None:
        """
        General function to apply any feature search method (RandomizedSearch, GridSearch, or BayesSearch).

        Args:
            search_class (class): The search class to use (RandomizedSearchCV, GridSearchCV, or BayesSearchCV).

        Raises:
            ValueError: If no model is provided.
        """
        if self.model is not None and self.search_class is not None:
            if self.search_class == "GridSearch":
                clf = search_class(
                    self.model,
                    self.valid_config.feature_engineering_dict,
                    cv=self.valid_config.cross_validation,
                )
            else:
                clf = search_class(
                    self.model,
                    self.valid_config.feature_engineering_dict,
                    n_iter=self.valid_config.number_iterations,
                    cv=self.valid_config.cross_validation,
                )
            if self.valid_config.standardized_df:
                x = (
                    pl.scan_csv(
                        self.valid_config.transformed_standardized_df_path_train_x
                        / self.valid_config.x_train_standardized_name
                    )
                    .collect()
                    .to_numpy()
                    .astype(np.float64)
                )
                y = (
                    pl.scan_csv(
                        self.valid_config.transformed_train_df_path_y
                        / self.valid_config.y_train_name
                    )
                    .collect()
                    .to_numpy()
                    .ravel()
                    .astype(np.int64)
                )

                search = clf.fit(x, y)

            elif self.valid_config.normalize_df:
                x = (
                    pl.scan_csv(
                        self.valid_config.transformed_normalized_df_path_train_x
                        / self.valid_config.x_train_normalized_name
                    )
                    .collect()
                    .to_numpy()
                )
                y = (
                    pl.scan_csv(
                        self.valid_config.transformed_train_df_path_y
                        / self.valid_config.y_train_name
                    )
                    .collect()
                    .to_numpy()
                    .ravel()
                )
                search = clf.fit(x, y)

            else:
                x = (
                    pl.scan_csv(
                        self.valid_config.transformed_train_df_path_x
                        / self.valid_config.x_train_name
                    )
                    .collect()
                    .to_pandas()
                    .to_numpy()
                )
                y = (
                    pl.scan_csv(
                        self.valid_config.transformed_train_df_path_y
                        / self.valid_config.y_train_name
                    )
                    .collect()
                    .to_pandas()
                    .to_numpy()
                    .ravel()
                )
                search = clf.fit(x, y)
            best_params = search.best_params_
            with open(
                self.valid_config.tuned_parameters_path
                / f"{self.valid_config.best_parameters_name}_{self.valid_config.feature_mode}_{self.valid_config.ML_type}.json",
                "w",
            ) as f:
                json.dump(best_params, f)
            logging.info(
                f"Best hyperparameters saved at {self.valid_config.tuned_parameters_path}/"
                f"{self.valid_config.best_parameters_name}_{self.valid_config.feature_mode}_{self.valid_config.ML_type}.json."
            )

        else:
            with ValueError as e:
                logging.error(
                    f"You need to specify a model of classification. You can find the the accepted types in "
                    f"the DataTransformationConfig class"
                    f", in the attribute {self.valid_config.ML_type} The error:\n{e}."
                )
                raise ValueError(e)

    def apply_feature_engineering(self) -> None:
        """
        Applies the specified feature engineering method.
        """
        mode = self.valid_config.feature_mode
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
            self.categorical_encoding()
            logging.info("Categorical Data Transformed and saved")

            self.split_train_test()
            logging.info("Split Data Transformed and saved")

            if self.valid_config.normalize_df:
                self.normalize()
                logging.info("Normalization applied.")

            elif self.valid_config.standardized_df:
                self.standardize()
                logging.info("Standardization applied.")
            else:
                logging.info("Skipped Normalization / Standardization.")
            logging.info(
                f"Stating Feature engineering with {self.valid_config.feature_mode}"
            )
            self.apply_feature_engineering()
            logging.info("Feature engineering applied.")

            logging.info("Pipeline Finished.")
        except Exception as e:
            logging.error(f"Pipeline failure at {e}")
            raise e
