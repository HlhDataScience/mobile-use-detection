"""This module holds the abstract class for the train and test. It is thought to be instanciated and wotk seemlessly with the other classes defined in this project."""

from abc import ABC, abstractmethod
from EDA_train_phase.src.abstractions.ABC_interfaces import (
IConfigModel,
IConfigurationLoader,
)


class BasicTrainer(ABC):
    """
    THis class is an abstraction to create custom trainer classes.
    This i the second part of the module and follows the instantiation of BasicPipeline class
    """

    def __init__(
        self,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        config_path: str,
        config_name: str,
        config_section: str = None,
    ):
        self.config_model = config_model
        self.config_data = config_loader.load(config_path, config_name)
        
        if config_section:
            if config_section in self.config_data:
                self.config_data = self.config_data[config_section]
            else:
                logging.error(f"Config section '{config_section}' not found.")
                raise KeyError(
                    f"Section '{config_section}' not found in the configuration."
                )


    @abstractmethod
    def train(self):
        """This mandatory function handles the training of the model"""
        pass

    @abstractmethod
    def test(self):
        """THis mandatory method handles testing"""
        pass

    @abstractmethod
    def run(self):
        """This mandatory methods runs the full trainer"""
