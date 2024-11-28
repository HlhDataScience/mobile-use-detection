""" THis module will be used to perform the train_test pipeline"""

import json
import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import mlflow
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from EDA_train_phase.src.abstractions.ABC_trainer import BasicTrainer
from EDA_train_phase.src.logging_functions.logger import setup_logging
from EDA_train_phase.src.validation_classes.validation_configurations import (
    TrainerConfig,
)
from EDA_train_phase.src.validation_classes.validation_interfaces import (
    HydraConfLoader,
    MLFlowTracker,
    PydanticConfigModel,
)

# CONSTANTS
LOG_FILE = Path("../logs/train_pipeline.log")
setup_logging(LOG_FILE)
CONFIG_PATH = "../../conf"


# INTERFACES LOADED

confi_model = PydanticConfigModel(config_model=TrainerConfig)
hydra_loader_conf = HydraConfLoader()
exp_tracker = MLFlowTracker()


class TrainerPipeline(BasicTrainer):
    """
    TrainerPipeline is a concrete implementation of the BasicTrainer abstract base class.
    It provides the functionality to perform a train-test pipeline, including scaling, training, evaluation, and logging
    with MLFlow integration.

    Attributes:
        normalized_df (bool): Indicates whether the data should be normalized.
        standardized_df (bool): Indicates whether the data should be standardized.

    Constructor Parameters:
        config_model (PydanticConfigModel): The configuration model for validation. Defaults to a predefined `PydanticConfigModel`.
        config_loader (HydraConfLoader): The configuration loader. Defaults to a predefined `HydraConfLoader`.
        experiment_tracker (MLFlowTracker): The MLFlow experiment tracker. Defaults to a predefined `MLFlowTracker`.
        config_path (str): The path to the configuration directory. Defaults to `../../conf`.
        config_name (str): The name of the configuration file. Defaults to `"config"`.
        config_section (str): The section of the configuration to load. Defaults to `"train_config"`.
        model (Any): The machine learning model to use. Defaults to `SVC()`.

    Methods:
        scaling_selection() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Selects and loads the appropriate scaled datasets (normalized or standardized) based on the configuration.

        train(self, x, y) -> Any:
            Trains the machine learning model with the provided training data and saves the model.

        eval(self, model, x, y) -> dict:
            Evaluates the model on the provided data and calculates various metrics.

        run(self):
            Executes the full training pipeline, including scaling selection, training, evaluation,
            parameter logging, and metric storage.

    Usage:
        Instantiate this class and call the `run` method to perform the end-to-end train-test pipeline.

    Raises:
        KeyError: If the specified configuration section is not found.
        pydantic.ValidationError: If the configuration validation fails.
    """

    def __init__(
        self,
        config_model: PydanticConfigModel = confi_model,
        config_loader: HydraConfLoader = hydra_loader_conf,
        experiment_tracker: MLFlowTracker = exp_tracker,
        config_path: str = CONFIG_PATH,
        config_name: str = "config",
        config_section: str = "train_config",
        model: Any = SVC(),
    ):

        super().__init__(
            config_model=config_model,
            config_loader=config_loader,
            experiment_tracker=experiment_tracker,
            config_path=config_path,
            config_name=config_name,
            config_section=config_section,
            model=model,
        )
        self.normalized_df = self.valid_config.normalized_df
        self.standardized_df = self.valid_config.standardized_df

    def scaling_selection(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Select the type of data based on if is needed to be normalized, standardized or None"""
        if self.normalized_df:
            x_train = (
                pl.scan_csv(self.valid_config.normalized_x_train).collect().to_numpy()
            )
            y_train = pl.scan_csv(self.valid_config.y_train).collect().to_numpy()
            x_test = (
                pl.scan_csv(self.valid_config.normalized_x_test).collect().to_numpy()
            )
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy()
        elif self.standardized_df:
            x_train = (
                pl.scan_csv(self.valid_config.standardized_x_train).collect().to_numpy()
            )
            y_train = pl.scan_csv(self.valid_config.y_train).collect().to_numpy()
            x_test = (
                pl.scan_csv(self.valid_config.standardized_x_test).collect().to_numpy()
            )
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy()
        else:
            x_train = pl.scan_csv(self.valid_config.x_train).collect().to_numpy()
            y_train = pl.scan_csv(self.valid_config.y_train).collect().to_numpy()
            x_test = pl.scan_csv(self.valid_config.x_test).collect().to_numpy()
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy()

        return x_train, x_test, y_train, y_test

    def train(self, x, y):
        """Placeholder at the moment"""
        parameters = json.load(self.valid_config.tuned_parameters)
        model_ = self.model(**parameters)
        model_.fit(x, y)
        joblib.dump(
            model_,
            f"{self.valid_config.model_path}/{self.valid_config.experiment_name}.joblib",
        )
        return model_

    def eval(self, model, x, y):
        """Placeholder at the moment"""
        y_proba = model.predict_proba(x)[:, 1]
        y_predicted = model.predict(x)
        return {
            "roc_auc": roc_auc_score(y, y_proba),
            "average_precision": average_precision_score(y, y_proba),
            "accuracy": accuracy_score(y, y_predicted),
            "precision": precision_score(y, y_predicted),
            "recall": recall_score(y, y_predicted),
            "f1": f1_score(y, y_predicted),
        }

    def run(self):
        """Placeholder at the moment"""
        try:
            x_train, x_test, y_train, y_test = self.scaling_selection()
            exp_id = self.experiment_tracker.get_or_create_experiment_id(
                name=self.valid_config.experiment_name
            )

            with mlflow.start_run(experiment_id=exp_id):
                ml_model = self.train(x_train, x_test)
                model_class = {"model_class": type(ml_model).__name__}
                model_parameters = {
                    f"{self.valid_config.experiment_name}__{k}": v
                    for k, v in ml_model.get_params().items()
                }
                model_and_parameters = model_class | model_parameters
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_parameters",
                    "w",
                ) as f:
                    json.dump(model_and_parameters, f)
                mlflow.log_param(
                    key=str(model_class.keys()), value=model_class.values()
                )
                mlflow.log_params(model_parameters)

                train_metrics = self.eval(model=ml_model, x=x_train, y=y_train)
                train_metrics_to_save = {
                    f"train__{k}": v for k, v in train_metrics.items()
                }
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_Train_metrics",
                    "w",
                ) as f:
                    json.dump(train_metrics_to_save, f)

                mlflow.log_metrics(train_metrics_to_save)

                test_metrics = self.eval(model=ml_model, x=x_test, y=y_test)
                test_metrics_to_save = {
                    f"test__{k}": v for k, v in test_metrics.items()
                }
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_Test_metrics",
                    "w",
                ) as f:
                    json.dump(test_metrics_to_save, f)
                mlflow.log_metrics(test_metrics_to_save)
        except Exception as e:
            logging.error(f"Pipeline failure at {e}")
            raise e
