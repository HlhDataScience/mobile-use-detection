"""This modules handle the wrappers for the validation and configuration yalm files."""

from typing import Any, Union

import omegaconf
import pandera
import pandera.polars
import pydantic

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


class OmegaConfLoader(IConfigurationLoader):
    """wrapper for OmegaConf"""

    def load(self, config_path: str) -> Any:
        return omegaconf.OmegaConf.load(config_path)
