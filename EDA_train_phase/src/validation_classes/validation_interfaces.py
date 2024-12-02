"""This module handle the wrappers for the validation and configuration yalm files."""

from typing import Any, Dict, Union

import mlflow
import pandera.polars
import pydantic
from hydra import compose
from mlflow import ActiveRun
from omegaconf import OmegaConf

from EDA_train_phase.src.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
    IExperimentTracker,
    IValidationModel,
)


class PanderaValidationModel(IValidationModel):
    """wrapper for pandera validation"""

    def __init__(
        self,
        validation_model: type[
            Union[pandera.polars.DataFrameModel | pandera.DataFrameModel]
        ],
    ):
        self.validation_model = validation_model

    def validate(self, dataframe: Any) -> Any:
        """Validates the dataframe schema using pandera"""
        return self.validation_model.validate(dataframe)


class PydanticConfigModel(IConfigModel):
    """wrapper for pydantic validation"""

    def __init__(self, config_model: type[pydantic.BaseModel]):
        self.config_model = config_model

    def parse(self, config_data: Any) -> Any:
        """Validates the configuration yalm file with pydantic"""
        return self.config_model(**config_data)


class HydraConfLoader(IConfigurationLoader):
    """wrapper for OmegaConf using Hydra"""

    def load(self, config_name: str) -> Dict:
        """Initialize Hydra and loads the configuration from yalm file."""
        hydra_config = compose(config_name=config_name)
        config_dict = OmegaConf.to_object(hydra_config)
        return config_dict


class MLFlowTracker(IExperimentTracker):
    """Wrapper for MLFlow experiment tracking"""

    def get_or_create_experiment_id(self, name: str) -> str:
        """Creates the ID of the experiment or use the current one."""
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            exp_id = mlflow.create_experiment(name)
            return exp_id
        return exp.experiment_id

    def initialize_experiment(self, experiment_id: Any) -> ActiveRun:
        return mlflow.start_run(experiment_id=experiment_id)

    def log_param(self, key: str, value: Union[int | float | str | Any]) -> None:
        """log the parameter into mlflow"""
        mlflow.log_param(key=key, value=value)

    def log_params(self, dictionary: Dict[str, Union[int | float | str | Any]]) -> None:
        """Log a dict of parameters into mlflow"""
        mlflow.log_params(params=dictionary)

    def log_metrics(
        self, dictionary: Dict[str, Union[int | float | str | Any]]
    ) -> None:
        """Log the metrics into mlflow"""
        mlflow.log_metrics(metrics=dictionary)

    def log_model(self, model: Any):
        """signature function override by log_model_signature"""
        pass

    def log_model_signature(self, model: Any, signature: Any, registered_model_name: str) -> None:  # type: ignore
        """logging the model into mlflow"""
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name=registered_model_name,
        )
