"""
Trainer Module

This module provides the abstract base class `BasicTrainer`, which serves as a foundation
for implementing custom training classes in machine learning workflows. It integrates
with configuration management, experiment tracking, and model evaluation components,
ensuring a consistent interface for training and evaluating machine learning models.

Classes:
- BasicTrainer: An abstract base class for creating custom trainer classes.
"""

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
    BasicTrainer Abstract Base Class

    The `BasicTrainer` class is an abstract base class designed to provide a
    structured approach for training and evaluating machine learning models.
    It requires subclasses to implement the core methods for training, evaluation,
    and running the training process. This class integrates with configuration
    management and experiment tracking systems to facilitate a streamlined
    training workflow.

    Attributes:
        experiment_tracker (IExperimentTracker): An instance responsible for tracking
            experiment details, metrics, and logging during the training and evaluation phases.
        config_model (IConfigModel): An instance that validates and parses configuration
            data for the training pipeline.
        config_data (dict): A dictionary containing configuration data loaded from a
            specified configuration file.
        model (Any): The machine learning model that will be trained and evaluated.

    Constructor Parameters:
        config_model (IConfigModel): The configuration model used for validation and parsing.
        config_loader (IConfigurationLoader): The configuration loader for fetching data
            from a configuration file.
        experiment_tracker (IExperimentTracker): The experiment tracker instance for monitoring
            the experiment.
        config_name (str): The name of the configuration file or section to be loaded.
        config_section (str, optional): A specific section in the configuration file to load.
            Defaults to `None`.
        model (Any, optional): The machine learning model. Defaults to `None`.

    Methods:
        train(self, x, y):
            Abstract method that must be implemented by subclasses to handle the training of the model.

        eval(self, x, y):
            Abstract method that must be implemented by subclasses to handle model evaluation.

        run(self):
            Abstract method that must be implemented by subclasses to execute the full training process.

    Raises:
        KeyError: If the specified configuration section is not found in the configuration data.
        pydantic.ValidationError: If the configuration validation fails.

    Usage:
        This class is not intended to be instantiated directly. Subclasses must implement
        the abstract methods (`train`, `eval`, `run`) to provide specific training logic.
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
        """
        Initializes the BasicTrainer instance.

        Args:
            config_model (IConfigModel): The configuration model used for validation and parsing.
            config_loader (IConfigurationLoader): The configuration loader for fetching data
                from a configuration file.
            experiment_tracker (IExperimentTracker): The experiment tracker instance for monitoring
                the experiment.
            config_name (str): The name of the configuration file or section to be loaded.
            config_section (str, optional): A specific section in the configuration file to load.
                Defaults to `None`.
            model (Any, optional): The machine learning model. Defaults to `None`.

        Raises:
            KeyError: If the specified configuration section is not found in the configuration data.
            pydantic.ValidationError: If the configuration validation fails.
        """
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
        """
        Trains the machine learning model using the provided input data.

        Args:
            x (Any): The input features for training the model.
            y (Any): The target labels corresponding to the input features.

        This method must be implemented by subclasses to define the specific training logic
        for the model.
        """
        pass

    @abstractmethod
    def eval(self, x, y):
        """
        Evaluates the performance of the machine learning model on the provided dataset.

        Args:
            x (Any): The input features for evaluation.
            y (Any): The target labels corresponding to the input features.

        This method must be implemented by subclasses to define the specific evaluation logic
        for the model.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Executes the complete training process, including training and evaluation.

        This method must be implemented by subclasses to orchestrate the entire training
        workflow, ensuring that the model is trained and evaluated appropriately.
        """
        pass
