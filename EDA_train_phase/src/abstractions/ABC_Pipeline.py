"""
This module defines the `BasicPipeline` abstract class, adhering to the SOLID principles
to provide a structured and extensible pipeline framework for data transformation and validation.

The pipeline integrates several key libraries:
- **Pandera**: For validating dataframes.
- **Pydantic**: For validating and parsing transformation configurations.
- **Hydra**: For managing configuration files (e.g., YAML) to automate pipeline setup.

Key Features:
- **Validation and Configuration Parsing**: Built-in mechanisms to ensure the integrity of input data and configurations using Pandera and Pydantic.
- **YAML Integration**: Designed to work with Hydra for configuration management, requiring a properly defined YAML file to streamline pipeline setup.
- **Reproducibility and Scalability**: Leverages configuration-driven design for reproducible and scalable workflows.
- **Extensibility**: Abstract methods enforce a standard pipeline structure, enabling custom implementations for specific tasks. Default methods allow for a more customizable pipeline usage.

Mandatory Requirement:
- The YAML configuration file must include a field named `original_datapath`, specifying the file path of the dataset to be validated. This is required for the `_validate_dataframe` method.

Usage:
--------
Subclasses should implement the abstract methods to define specific pipeline behaviors,
such as data transformations, scaling, and feature engineering.

Example Subclass:

class MyPipeline(BasicPipeline):
    def custom_validate(self):
        # Custom validation logic here
        pass

    def categorical_to_numerical(self):
        # Implementation here
        pass

    def run(self):
        # Execute pipeline
        pass
"""

import logging
from abc import ABC, abstractmethod
from typing import Union

import omegaconf
import pandera
import polars as pl
import pydantic
from pandera import polars


class BasicPipeline(ABC):
    """
    Abstract base class for defining data pipelines with built-in support for validation
    and configuration management using Hydra, Pydantic, and Pandera.

    This class is designed to:
    - Validate input dataframes using Pandera's validation models.
    - Parse and validate pipeline configurations using Pydantic models.
    - Automate the management of configurations through Hydra (e.g., YAML files).
    - Enforce a structured and extensible workflow through abstract methods.

    Constructor:
    -------------
    __init__(
        validation_model: Union[pandera.polars.DataFrameModel, pandera.DataFrameModel],
        config_model: type[pydantic.BaseModel],
        config_data: omegaconf.DictConfig,
        apply_custom_function: bool
    )

    Args:
    -----
    - validation_model (Union[pandera.polars.DataFrameModel, pandera.DataFrameModel]):
        A Pandera model for validating input dataframes.
    - config_model (type[pydantic.BaseModel]):
        A Pydantic model subclass to parse and validate pipeline configuration.
    - config_data (omegaconf.DictConfig):
        Configuration data loaded via Hydra or equivalent YAML management tool.
        **Note**: This must include a field `original_datapath` specifying the dataset path.
    - apply_custom_function (bool):
        Determines whether to use a custom validation function (`custom_validate`).

    Key Features:
    -------------
    - **Built-in Validation**: Ensures configuration validity and data integrity
      through Pydantic and Pandera, reducing manual errors.
    - **Hydra Integration**: Simplifies configuration handling for scalable and
      reproducible workflows.
    - **Extensibility**: Abstract methods define a flexible framework for custom
      pipeline logic, enabling developers to implement transformations, scaling,
      and other operations.
    - **Logging**: Provides informative logging during validation and configuration
      parsing to aid debugging.
    - **Mandatory YAML Field**: The `config_data` must include `original_datapath`
      to specify the dataset's file path. This is essential for the `_validate_dataframe` method.

    Additional Elements:
    --------------------
    - **Error Handling and Logging**: Comprehensive error handling ensures that the
      pipeline fails gracefully during validation and configuration parsing.
    - **Reproducibility and Scalability**: Using configuration-driven design (Hydra
      and Pydantic) allows the pipeline to adapt to new workflows with minimal code changes.
    - **Integration Points**: The class is compatible with libraries like Polars, Pandera,
      and Hydra, making it easy to integrate with existing data processing tools.
    - **Target Audience**: Designed for data scientists and ML engineers seeking
      validation, configuration management, and scalable workflows.

    Default Methods:
    -----------------
    Subclasses may implement, depending on the needs, the following methods:

    - `categorical_to_numerical`: Conversion of categorical features to numerical.
    - `split_train_test`: Splitting data into training and test sets.
    - `scaling`: Scaling data features.
    - `normalize`: Normalizing data features.
    - `standardize`: Standardizing data features.
    - `_apply_feature_search`: Applying feature search techniques.
    - `apply_feature_engineering`: Implementing feature engineering.

    WARNING: calling these methods without implementation will raise a NotImplementedError.
    Abstract Methods:
    -----------------
    Subclasses must implement the following:
    - `custom_validate`: Custom logic for data validation.
    - `run`: Executing the pipeline end-to-end.
    """

    def __init__(
        self,
        validation_model: Union[pandera.polars.DataFrameModel | pandera.DataFrameModel],
        config_model: type[pydantic.BaseModel],
        config_data: omegaconf.DictConfig,
        apply_custom_function: bool,
    ):

        if not issubclass(config_model, pydantic.BaseModel):
            raise TypeError("'Config' must be a subclass of 'pydantic.BaseModel'")

        self.validation_model = validation_model
        self.config_model = config_model
        self.config_data = config_data
        self.apply_custom_function = apply_custom_function

        try:
            self.valid_config = self.config_model(**self.config_data)
            logging.info(
                "Data Pipeline configuration was successfully loaded and validated"
            )
        except pydantic.ValidationError as e:
            logging.error(f"Validation of the data configuration failed at:\n{e}")

        if self.apply_custom_function:
            try:
                self.custom_validate()
                logging.info("Validation of the dataframe was successful!")
            except ValueError as e:
                logging.error(f"The validation failed at:\n{e}")

        else:
            try:
                self._validate_dataframe()
                logging.info("Validation of the dataframe was sucessful!")
            except ValueError as e:
                logging.error(f"The validation of the Dataframe failed at:\n{e}")

    def _validate_dataframe(self) -> None:
        """Default implementation of the pandera.polars validation using csv."""
        pl.scan_csv(self.valid_config.original_datapath).pipe(
            self.validation_model.validate
        )

    @abstractmethod
    def custom_validate(self):
        """Custom mandatory validation method."""
        pass

    def categorical_to_numerical(self):
        """Default implementation for categorical to numerical transformation."""
        logging.warning("categorical_to_numerical method is not implemented")
        raise NotImplementedError()

    def split_train_test(self):
        """Default implementation for train-test split."""
        logging.warning("split_train_test method is not implemented")
        raise NotImplementedError()

    def scaling(self):
        """Default implementation for scaling."""
        logging.warning("scaling method is not implemented")
        raise NotImplementedError()

    def normalize(self):
        """Default implementation for normalize"""
        logging.warning("normalize method is not implemented")
        raise NotImplementedError()

    def standardize(self):
        """Default implementation for standardize."""
        logging.warning("standardize method is not implemented")
        raise NotImplementedError()

    def _apply_feature_search(self):
        """Default implementation for _apply_feature_search."""
        logging.warning("_apply_feature_search method is not implemented")
        raise NotImplementedError()

    def apply_feature_engineering(self):
        """Default implementation for apply_feature_engineering."""
        logging.warning("apply_feature_engineering method is not implemented")
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        """Mandatory implementation for the run method that orchestrates the whole pipeline class"""
        pass
