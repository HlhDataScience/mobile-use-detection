"""
This module defines the `BasicPipeline` abstract class, adhering to the SOLID principles
to provide a structured and extensible pipeline framework for DataTrain transformation and validation.

The pipeline integrates several key libraries:
- **Pandera**: For validating dataframes.
- **Pydantic**: For validating and parsing transformation configurations.
- **Hydra**: For managing configuration files (e.g., YAML) to automate pipeline setup.

Key Features:
-------------
- **Validation and Configuration Parsing**: Built-in mechanisms to ensure the integrity of input DataTrain and configurations using Pandera and Pydantic.
- **YAML Integration**: Designed to work with Hydra for configuration management, requiring a properly defined YAML file to streamline pipeline setup.
- **Selective Sub-Configuration Parsing**: Supports loading and validating specific sections of the YAML configuration file using the `config_section` parameter.
- **Reproducibility and Scalability**: Leverages configuration-driven design for reproducible and scalable workflows.
- **Extensibility**: Abstract methods enforce a standard pipeline structure, enabling custom implementations for specific tasks. Default methods allow for a more customizable pipeline usage.

Mandatory Requirement:
----------------------
- The YAML configuration file must include a field named `original_datapath`, specifying the file path of the dataset to be validated. This is required for the `_validate_dataframe` method.
- If the `config_section` argument is provided, the specified section must exist in the YAML configuration. A `KeyError` will be raised if the section is missing.

Usage:
------
Subclasses should implement the abstract methods to define specific pipeline behaviors,
such as DataTrain transformations, scaling, and feature engineering.

Example Subclass:

```python
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
´´´
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from types import MethodType
from typing import List

import polars as pl
import pydantic

from Phases.SrcTrain.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
    IValidationModel,
)


class BasicPipeline(ABC):
    """
    Abstract base class for defining DataTrain pipelines with built-in support for validation
    and configuration management using abstraction layers for various libraries.

    This class is designed to:
    - Validate input dataframes using an abstraction of validation ModelsTrain.
    - Parse and validate pipeline configurations using an abstraction of configuration ModelsTrain.
    - Automate the management of configurations through an abstraction of configuration loaders.
    - Enforce a structured and extensible workflow through abstract methods.

    Constructor:
    -------------
    __init__(
        validation_model: IValidationModel,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        config_path: str,
        config_name: str,
        apply_custom_function: bool,
        config_section: str = None
    )

    Args:
    -----
    - validation_model (IValidationModel):
        An abstraction for validating input dataframes.
    - config_model (IConfigModel):
        An abstraction for parsing and validating pipeline configuration.
    - config_loader (IConfigurationLoader):
        An abstraction for loading configuration DataTrain from specified paths.
    - config_path (str):
        The path to the configuration file to be loaded.
    - config_name (str):
        The name of the main configuration file (e.g., 'main').
    - apply_custom_function (bool):
        Determines whether to use a custom validation function (`custom_validate`).
    - config_section (str, optional):
        Specifies a sub-configuration section (e.g., `transformation_config`) within the main configuration.
        If provided, the corresponding section will be extracted and validated. Defaults to `None`.

    Key Features:
    -------------
    - **Built-in Validation**: Ensures configuration validity and DataTrain integrity
      through the use of validation abstractions, reducing manual errors.
    - **Flexible Configuration Handling**: Allows for dynamic selection of specific sub-configurations
      within a larger configuration file using the `config_section` parameter.
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
    - **Reproducibility and Scalability**: Using configuration-driven design allows the pipeline
      to adapt to new workflows with minimal code changes.
    - **Integration Points**: The class is compatible with libraries through abstractions,
      making it easy to integrate with existing DataTrain processing tools.
    - **Target Audience**: Designed for DataTrain scientists and ML engineers seeking
      validation, configuration management, and scalable workflows.

    Default Methods:
    -----------------
    Subclasses may implement, depending on the needs, the following methods:

    - `categorical_encoding`: Conversion of categorical features to numerical.
    - `split_train_test`: Splitting DataTrain into training and test sets.
    - `scaling`: Scaling DataTrain features.
    - `normalize`: Normalizing DataTrain features.
    - `standardize`: Standardizing DataTrain features.
    - `_apply_feature_search`: Applying feature search techniques.
    - `apply_feature_engineering`: Implementing feature engineering.

    WARNING: Calling these methods without implementation will raise a NotImplementedError.

    Abstract Methods:
    -----------------
    Subclasses must implement the following:
    - `custom_validate`: Custom logic for DataTrain validation.
    - `run`: Executing the pipeline end-to-end.

    Updates:
    --------
    - **`config_section` Argument**: Enables loading and validating a specific section
      of the configuration file. This is useful for modular pipelines where each phase
      (e.g., SrcEda, transformation, training) has its own configuration.
    - **Error Handling for Missing Sections**: Raises a `KeyError` if the specified
      `config_section` does not exist in the configuration file.
    """

    def __init__(
        self,
        validation_model: IValidationModel,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        config_name: str,
        apply_custom_function: bool,
        config_section: str,
    ):
        self.validation_model = validation_model
        self.config_model = config_model
        self.config_data = config_loader.load(config_name)
        self.apply_custom_function = apply_custom_function
        self.functions: List[Callable] = []
        if config_section:
            if config_section in self.config_data:
                self.config_data = self.config_data[config_section]
            else:
                logging.error(f"Config section '{config_section}' not found.")
                raise KeyError(
                    f"Section '{config_section}' not found in the configuration."
                )

        try:
            self.valid_config = self.config_model.parse(self.config_data)
            logging.info(
                "Data Pipeline configuration was successfully loaded and validated"
            )
        except pydantic.ValidationError as e:
            logging.error(f"Validation of the DataTrain configuration failed at:\n{e}")

        if self.apply_custom_function:
            try:
                self.custom_validate()
                logging.info("Validation of the dataframe was successful!")
            except ValueError as e:
                logging.error(f"The validation failed at:\n{e}")
        else:
            try:
                self._validate_dataframe()
                logging.info("Validation of the dataframe was successful!")
            except ValueError as e:
                logging.error(f"The validation of the Dataframe failed at:\n{e}")

    def _validate_dataframe(self) -> None:
        """Default implementation of the pandera validation."""
        pl.scan_csv(self.valid_config.original_datapath).pipe(
            self.validation_model.validate
        )

    def custom_validate(self):
        """Custom  validation method."""
        raise NotImplementedError

    @classmethod
    def from_functions(
        cls,
        validation_model: IValidationModel,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        config_name: str,
        apply_custom_function: bool,
        config_section: str,
        functions_constructor: List[Callable],
    ) -> "BasicPipeline":
        """
        Creates a BasicPipeline instance and allows customization through provided functions.

        Args:
            validation_model (IValidationModel): Validation model for the pipeline.
            config_model (IConfigModel): Configuration model for parsing and validation.
            config_loader (IConfigurationLoader): Loader for configuration data.
            config_name (str): Name of the configuration file.
            apply_custom_function (bool): Whether to apply custom validation logic.
            config_section (str): Section of the configuration to load.
            functions_constructor (Tuple[Callable, ...]): Additional functions to customize behavior.

        Returns:
            BasicPipeline: An initialized instance of the BasicPipeline.
        """
        # Create the pipeline instance
        instance = cls(
            validation_model=validation_model,
            config_model=config_model,
            config_loader=config_loader,
            config_name=config_name,
            apply_custom_function=apply_custom_function,
            config_section=config_section,
        )

        # Store the functions to be executed later
        instance.functions = functions_constructor

        # Dynamically assign the run method to execute the functions in sequence
        def run(self):
            """
            Executes the pipeline by running the functions in sequence.
            """
            logging.info("Pipeline is starting.")
            for function in self.functions:
                logging.info(f"Running function: {function.__name__}")
                function(self)
            logging.info("Pipeline execution finished.")

        # Override the run method for this instance
        instance.run = MethodType(run, instance)

        return instance

    @abstractmethod
    def run(self):
        """
        Mandatory implementation for the run method that orchestrates the whole pipeline class.
        All the functions must be run through this method.
        """
        pass
