"""
ABC Validations Module

This module provides abstract base classes (ABCs) for various components related to
data validation, configuration management, and experiment tracking. The use of ABCs
ensures that any concrete implementation of these components adheres to a defined
interface, promoting consistency and reliability across different implementations.

Classes:
- IValidationModel: Defines the interface for validating dataframes.
- IConfigModel: Defines the interface for parsing configuration data.
- IConfigurationLoader: Defines the interface for loading configurations from a specified path.
- IExperimentTracker: Defines the interface for tracking experiments, including logging parameters, metrics, and models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class IValidationModel(ABC):
    """
    Interface for Validation Models

    This abstract base class defines the contract for validation models that
    validate dataframes. Implementations must provide the logic for validating
    the structure and contents of a dataframe.

    Methods:
    - validate: Validates the given dataframe and returns the result of the validation.
    """

    @abstractmethod
    def validate(self, dataframe: Any) -> Any:
        """
        Validates the given dataframe.

        Args:
            dataframe (Any): The dataframe to be validated.

        Returns:
            Any: The result of the validation process, which may include
            validation status, errors, or transformed data.
        """
        pass


class IConfigModel(ABC):
    """
    Interface for Configuration Models

    This abstract base class defines the contract for configuration models
    that parse configuration data. Implementations must provide the logic
    for interpreting and validating configuration inputs.

    Methods:
    - parse: Parses the configuration data and returns the structured output.
    """

    @abstractmethod
    def parse(self, config_data: Any) -> Any:
        """
        Parses the configuration data.

        Args:
            config_data (Any): The configuration data to be parsed.

        Returns:
            Any: The structured output of the parsed configuration.
        """
        pass


class IConfigurationLoader(ABC):
    """
    Interface for Configuration Loaders

    This abstract base class defines the contract for loaders that retrieve
    configuration data from a specified path. Implementations must provide
    the logic for loading configuration files or data sources.

    Methods:
    - load: Loads the configuration from the specified path.
    """

    @abstractmethod
    def load(self, config_name: str) -> Any:
        """
        Loads the configuration from the specified path.

        Args:
            config_name (str): The name or path of the configuration to load.

        Returns:
            Any: The loaded configuration data.
        """
        pass


class IExperimentTracker(ABC):
    """
    Interface for Experiment Tracking

    This abstract base class defines the contract for tracking experiments.
    Implementations must provide methods for initializing experiments and
    logging parameters, metrics, and models.

    Methods:
    - get_or_create_experiment_id: Retrieves or creates an experiment ID.
    - initialize_experiment: Initializes the tracking for a given experiment.
    - log_param: Logs a single parameter for the experiment.
    - log_params: Logs multiple parameters for the experiment.
    - log_metrics: Logs metrics for the experiment.
    - log_model: Logs the model used in the experiment.
    """

    @abstractmethod
    def get_or_create_experiment_id(self, name: str):
        """
        Retrieves or creates an experiment ID based on the given name.

        Args:
            name (str): The name of the experiment.

        Returns:
            Any: The unique identifier for the experiment.
        """
        pass

    @abstractmethod
    def initialize_experiment(self, experiment_id: Any):
        """
        Initializes the experiment tracking with the provided experiment ID.

        Args:
            experiment_id (Any): The unique identifier for the experiment.
        """
        pass

    @abstractmethod
    def log_param(self, key: str, value: Union[int, float, str, Any]) -> None:
        """
        Logs a single parameter for the experiment.

        Args:
            key (str): The name of the parameter.
            value (Union[int, float, str, Any]): The value of the parameter.
        """
        pass

    @abstractmethod
    def log_params(self, dictionary: Dict[str, Union[int, float, str, Any]]) -> None:
        """
        Logs multiple parameters for the experiment.

        Args:
            dictionary (Dict[str, Union[int, float, str, Any]]): A dictionary of parameters to log.
        """
        pass

    @abstractmethod
    def log_metrics(self, dictionary: Dict[str, float]) -> None:
        """
        Logs metrics for the experiment.

        Args:
            dictionary (dictionary: Dict[str, float]): A dictionary of metrics to log.
        """
        pass

    @abstractmethod
    def log_model(self, model: Any) -> None:
        """
        Logs the model used in the experiment.

        Args:
            model (Any): The model instance to log.
        """
        pass
