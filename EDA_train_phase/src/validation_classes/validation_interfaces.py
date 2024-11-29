"""This module handle the wrappers for the validation and configuration yalm files."""

from typing import Any, Dict, Union

import mlflow
import pandera.polars
import pydantic
from hydra import compose
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
