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
from typing import List

import polars as pl
import pydantic

from src.train.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
    IValidationModel,
)


class BasicPipeline(ABC):
    """
    Abstract base class for building scalable and extensible data pipelines with
    validation, configuration management, and structured workflows.

    This class provides a foundation for implementing data pipelines that include:
    - Input validation using a specified validation model.
    - Configuration parsing and validation with a configuration model.
    - Dynamic configuration management through a configuration loader.
    - Customizable and reusable pipeline workflows.

    Constructor:
    -------------
    __init__(
        validation_model: IValidationModel,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        config_name: str,
        apply_custom_function: bool,
        config_section: str = None
    )

    Args:
    -----
    - validation_model (IValidationModel):
        A model responsible for validating input dataframes.
    - config_model (IConfigModel):
        A model for parsing and validating configuration data.
    - config_loader (IConfigurationLoader):
        Loader for retrieving configuration data from external sources.
    - config_name (str):
        Name of the configuration file to be loaded.
    - apply_custom_function (bool):
        If `True`, uses the `custom_validate` method for custom validation logic.
        If `False`, uses the default `_validate_dataframe` method.
    - config_section (str, optional):
        Name of a subsection within the configuration to be loaded and validated.
        If provided, only the specified section is used. Defaults to `None`.

    Attributes:
    -----------
    - functions (List[Callable]):
        A list of callable functions that define the sequence of operations for the pipeline.
        These functions are executed in order when `run_from_functions` is called.
        This attribute is optional and should be populated in subclasses or externally
        before executing the pipeline.

    Key Features:
    -------------
    - **Validation**: Supports both default and custom validation methods to ensure
      data and configuration integrity.
    - **Configuration Handling**: Allows dynamic loading and validation of configurations,
      including the ability to focus on specific subsections.
    - **Extensibility**: Abstract methods enforce implementation of custom pipeline logic
      while enabling reuse of common functionalities.
    - **Orchestrated Execution**: Allows users to define a sequence of operations
      using `functions`, enabling modular and reusable workflows.
    - **Logging**: Provides detailed logs for each step, aiding debugging and monitoring.

    Methods:
    --------
    - `_validate_dataframe`: Default validation logic using the provided validation model.
    - `custom_validate`: Abstract method for implementing custom validation logic.
    - `run_from_functions`: Executes the pipeline by running the `functions` in sequence.
    - `run`: Abstract method that orchestrates the pipeline workflow.

    Usage:
    ------
    1. Define or extend the `BasicPipeline` class and implement the `custom_validate` and `run` methods.
    2. Populate the `functions` attribute with a list of callable functions.
       Each function must accept the pipeline instance as its argument.
    3. Use the `run_from_functions` method to execute the functions in order
       or implement custom orchestration logic in the `run` method.

    Example:
    --------
    ```python
    class MyPipeline(BasicPipeline):
        def custom_validate(self):
            # Custom validation logic
            pass

        def run(self):
            # Orchestrate the pipeline
            self.run_from_functions()

    pipeline = MyPipeline(
        validation_model=my_validation_model,
        config_model=my_config_model,
        config_loader=my_config_loader,
        config_name="pipeline_config.yaml",
        apply_custom_function=True,
    )

    pipeline.functions = [step1, step2, step3]  # Define pipeline steps
    pipeline.run()  # Execute the pipeline
    ```

    Abstract Methods:
    -----------------
    - `custom_validate`: To be implemented for specific validation logic.
    - `run`: To be implemented for orchestrating the pipeline workflow.

    Error Handling:
    ---------------
    - Logs and raises a `KeyError` if the specified `config_section` is missing.
    - Logs validation errors for configuration and data issues.
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
        """Custom validation method."""
        raise NotImplementedError

    def run_from_functions(self):
        """
        Executes the pipeline by running the functions in sequence.
        This method can be overridden by subclasses if needed.
        """
        logging.info("Pipeline is starting.")
        for function in self.functions:
            logging.info(f"Running function: {function.__name__}")
            function(self)
        logging.info("Pipeline execution finished.")

    @abstractmethod
    def run(self):
        """
        Mandatory implementation for the run method that orchestrates the whole pipeline class.
        All the functions must be run through this method.
        """
        pass
