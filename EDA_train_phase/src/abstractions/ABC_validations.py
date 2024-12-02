"""module for ABC validation"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class IValidationModel(ABC):
    """Wrapper for Validation dataframe models"""

    @abstractmethod
    def validate(self, dataframe: Any) -> Any:
        """Validates the given dataframe."""
        pass


class IConfigModel(ABC):
    """wrapper for configuration management"""

    @abstractmethod
    def parse(self, config_data: Any) -> Any:
        """Parses the configuration data."""
        pass


class IConfigurationLoader(ABC):
    """Loads the configuration from the specified path."""

    @abstractmethod
    def load(self, config_name: str) -> Any:
        """Loads the configuration from the specified path."""
        pass


class IExperimentTracker(ABC):
    """Wrapper for experiment tracking"""

    @abstractmethod
    def get_or_create_experiment_id(self, name: str):
        """This function handle the creation of the experiment id tracking by the interface"""
        pass

    @abstractmethod
    def initialize_experiment(self, experiment_id: Any):
        """Wrapper around starting the experiment"""
        pass

    @abstractmethod
    def log_param(self, key: str, value: Union[int | float | str | Any]) -> None:
        """Handles the log parameter function"""
        pass

    @abstractmethod
    def log_params(self, dictionary: Dict[str, Union[int | float | str | Any]]) -> None:
        """Handles the log parameters function"""
        pass

    @abstractmethod
    def log_metrics(
        self, dictionary: Dict[str, Union[int | float | str | Any]]
    ) -> None:
        """Handles the metrics logging"""

    @abstractmethod
    def log_model(self, model: Any) -> None:
        """Defines the log model function"""
        pass
