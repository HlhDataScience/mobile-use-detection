"""This module holds the abstract class for the train and test. It is thought to be instanciated and wotk seemlessly with the other classes defined in this project."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import pydantic

from EDA_train_phase.src.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
    IExperimentTracker,
)


class BasicTrainer(ABC):
    """
    BasicTrainer is an abstract base class (ABC) designed to serve as a foundation for creating custom trainer classes.
    It integrates seamlessly with other components of the project and follows the instantiation of the BasicPipeline class.

    Attributes:
        experiment_tracker (IExperimentTracker): Tracks experiment details and metrics during training and evaluation.
        config_model (IConfigModel): Validates and parses configuration data for the training pipeline.
        config_data (dict): Configuration data loaded from a specified file and section.
        model (Any): The machine learning model to be trained and evaluated.

    Constructor Parameters:
        config_model (IConfigModel): The configuration model used to validate the loaded configuration.
        config_loader (IConfigurationLoader): The configuration loader used to fetch data from a configuration file.
        experiment_tracker (IExperimentTracker): The experiment tracker instance for monitoring the experiment.
        config_path (str): Path to the configuration file.
        config_name (str): Name of the configuration file or section to be loaded.
        config_section (str, optional): Specific section in the configuration file to load. Defaults to `None`.
        model (Any, optional): The machine learning model. Defaults to `None`.

    Methods:
        train(self, x, y):
            Abstract method to handle the training of the model. Must be implemented by subclasses.

        eval(self, x, y):
            Abstract method to handle model evaluation. Must be implemented by subclasses.

        run(self):
            Abstract method to execute the full training process. Must be implemented by subclasses.

    Raises:
        KeyError: If the specified configuration section is not found in the configuration data.
        pydantic.ValidationError: If the configuration validation fails.

    Usage:
        This class is not meant to be instantiated directly. Instead, it should be subclassed,
        and the abstract methods (`train`, `eval`, `run`) must be implemented in the subclass.
    """

    def __init__(
        self,
        config_model: IConfigModel,
        config_loader: IConfigurationLoader,
        experiment_tracker: IExperimentTracker,
        config_name: str,
        config_section: str = None,
        model: Any = None,
    ):
        self.experiment_tracker = experiment_tracker
        self.config_model = config_model
        self.config_data = config_loader.load(config_name)
        self.model = model
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
            logging.error(f"Validation of the data configuration failed at:\n{e}")

    @abstractmethod
    def train(self, x, y):
        """This mandatory function handles the training of the model"""
        pass

    @abstractmethod
    def eval(self, x, y):
        """THis mandatory method handles testing"""
        pass

    @abstractmethod
    def run(self):
        """This mandatory methods runs the full trainer"""
