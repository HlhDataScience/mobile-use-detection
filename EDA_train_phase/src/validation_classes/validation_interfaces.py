"""This modules handle the wrappers for the validation and configuration yalm files."""

from typing import Any, Union

import omegaconf
import pandera
import pandera.polars
import pydantic
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from EDA_train_phase.src.abstractions.ABC_validations import (
    IConfigModel,
    IConfigurationLoader,
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
        return self.validation_model.validate(dataframe)


class PydanticConfigModel(IConfigModel):
    """wrapper for pydantic validation"""

    def __init__(self, config_model: type[pydantic.BaseModel]):
        self.config_model = config_model

    def parse(self, config_data: Any) -> Any:
        return self.config_model(**config_data)


class HydraConfLoader(IConfigurationLoader):
    """wrapper for OmegaConf using Hydra"""

    def load(self, config_path: str, config_name: str) -> Any:
        if GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()

        initialize(config_path=config_path, job_name="load_config")

        hydra_config = compose(config_name=config_name)
        config_dict = OmegaConf.to_object(hydra_config)
        return config_dict
