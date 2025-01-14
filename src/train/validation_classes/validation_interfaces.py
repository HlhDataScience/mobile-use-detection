"""
This module provides wrappers for validation and configuration management using
various libraries such as Pandera, Pydantic, Hydra, and MLFlow. It defines
interfaces for validation ModelsTrain, configuration loaders, and experiment tracking.
"""

from enum import Enum
from typing import Any, Dict, Union

import mlflow
import pandera.polars
import pydantic
from hydra import compose
from mlflow import ActiveRun
from omegaconf import OmegaConf

from src.train.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
    IExperimentTracker,
    IValidationModel,
)


class PanderaValidationModel(IValidationModel):
    """A wrapper for validating dataframes using Pandera."""

    def __init__(
        self,
        validation_model: type[
            Union[pandera.polars.DataFrameModel | pandera.DataFrameModel]
        ],
    ):
        """
        Initializes the PanderaValidationModel with a specified validation ModelsProduction.

        Args:
            validation_model (type): A Pandera DataFrameModel or Polars DataFrameModel type used for validation.
        """
        self.validation_model = validation_model

    def validate(self, dataframe: Any) -> Any:
        """
        Validates the schema of the provided dataframe using the Pandera validation ModelsProduction.

        Args:
            dataframe (Any): The dataframe to be validated.

        Returns:
            Any: The validated dataframe, or raises an error if validation fails.
        """
        return self.validation_model.validate(dataframe)


class PydanticConfigModel(IConfigModel):
    """A wrapper for validating configuration DataTrain using Pydantic."""

    def __init__(self, config_model: type[pydantic.BaseModel]):
        """
        Initializes the PydanticConfigModel with a specified Pydantic ModelsProduction.

        Args:
            config_model (type): A Pydantic BaseModel type used for configuration validation.
        """
        self.config_model = config_model

    def parse(self, config_data: Any) -> Any:
        """
        Validates and parses the given configuration DataTrain using the Pydantic ModelsProduction.

        Args:
            config_data (Any): The configuration DataTrain to be validated and parsed.

        Returns:
            Any: An instance of the Pydantic ModelsProduction populated with the validated configuration DataTrain.

        Raises:
            ValidationError: If the provided configuration DataTrain does not conform to the ModelsProduction.
        """
        return self.config_model(**config_data)


class HydraConfLoader(IConfigurationLoader):
    """A wrapper for loading configuration files using Hydra and OmegaConf."""

    def load(
        self, config_name: str
    ) -> Dict[str | bytes | int | Enum | float | bool, Any] | list | None | str:
        """
        Loads the configuration from a YAML file using Hydra.

        Args:
            config_name (str): The name of the configuration to be loaded.

        Returns:
            Dict: A dictionary representation of the loaded configuration.
        """
        hydra_config = compose(config_name=config_name)
        config_dict = OmegaConf.to_object(hydra_config)
        return config_dict


class MLFlowTracker(IExperimentTracker):
    """A wrapper for tracking experiments using MLFlow."""

    def get_or_create_experiment_id(self, name: str) -> str:
        """
        Retrieves the ID of an existing experiment or creates a new one if it does not exist.

        Args:
            name (str): The name of the experiment.

        Returns:
            str: The ID of the experiment.
        """
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
            return exp_id
        return exp.experiment_id

    def initialize_experiment(self, experiment_id: Any) -> ActiveRun:
        """
        Starts a new MLFlow run for the specified experiment ID.

        Args:
            experiment_id (Any): The ID of the experiment to initialize.

        Returns:
            ActiveRun: An active run object for the experiment.
        """
        return mlflow.start_run(experiment_id=experiment_id)

    def log_param(self, key: str, value: Union[int | float | str | Any]) -> None:
        """
        Logs a single parameter to MLFlow.

        Args:
            key (str): The name of the parameter.
            value (Union[int, float, str, Any]): The value of the parameter to log.
        """
        mlflow.log_param(key=key, value=value)

    def log_params(self, dictionary: Dict[str, Union[int | float | str | Any]]) -> None:
        """
        Logs multiple parameters to MLFlow.

        Args:
            dictionary (Dict[str, Union[int, float, str, Any]]): A dictionary of parameters to log.
        """
        mlflow.log_params(params=dictionary)

    def log_metrics(self, dictionary: Dict[str, float]) -> None:
        """
               Logs MetricsTrain to MLFlow.

               Args:
        dictionary (Dict[str,  float]): A dictionary of MetricsTrain to log.
        """
        mlflow.log_metrics(metrics=dictionary)

    def log_model(self, model: Any):
        """
        Logs a ModelsProduction to MLFlow. This method is intended to be overridden by
        log_model_signature for specific ModelsProduction logging implementations.

        Args:
            model (Any): The ModelsProduction to log.
        """
        pass

    @staticmethod
    def log_model_signature(model: Any, signature: Any, registered_model_name: str) -> None:  # type: ignore
        """
        Logs a ModelsProduction along with its signature to MLFlow.

        Args:
            model (Any): The ModelsProduction to log.
            signature (Any): The signature of the ModelsProduction.
            registered_model_name (str): The name under which the ModelsProduction is registered in MLFlow.
        """
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-ModelsProduction",
            signature=signature,
            registered_model_name=registered_model_name,
        )
