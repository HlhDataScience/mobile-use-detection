"""module for ABC validation"""

from abc import ABC, abstractmethod
from typing import Any


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
    def load(self, config_path: str, config_name: str) -> Any:
        """Loads the configuration from the specified path."""
        pass
