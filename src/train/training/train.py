""" THis module will be used to perform the train_test pipeline"""

import json
import logging
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import polars as pl
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from typing_extensions import override

from src.train.abstractions.ABC_trainer import BasicTrainer
from src.train.validation_classes.validation_interfaces import (
    HydraConfLoader,
    MLFlowTracker,
    PydanticConfigModel,
)


class TrainerPipeline(BasicTrainer):
    """
    TrainerPipeline is a concrete implementation of the BasicTrainer abstract base class.
    It orchestrates the end-to-end training pipeline, including DataTrain scaling, ModelsProduction training, evaluation,
    and logging MetricsTrain with MLFlow integration. The pipeline is designed to adhere to SOLID principles
    and supports configuration-driven workflows.

    Attributes:
        normalized_df (bool): Indicates whether the training and testing DataTrain should be normalized.
        standardized_df (bool): Indicates whether the training and testing DataTrain should be standardized.

    Constructor Parameters:
        config_model (PydanticConfigModel): The Pydantic configuration ModelsProduction for validation and consistency checks.
        config_loader (HydraConfLoader): The Hydra-based configuration loader for managing YAML configurations.
        experiment_tracker (MLFlowTracker): An MLFlow tracker for managing experiments and runs.
        config_path (str): Path to the directory containing configuration files. Defaults to `../../ConfTrain`.
        config_name (str): Name of the configuration file. Defaults to `"config"`.
        config_section (str): Section of the configuration file to load. Defaults to `"train_config"`.
        ModelsProduction (BaseEstimator): The scikit-learn-compatible machine learning ModelsProduction. Defaults to `SVC()`.

    Methods:
        scaling_selection() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Selects and loads scaled datasets (normalized, standardized, or raw) based on configuration settings.

        train(x, y) -> BaseEstimator:
            Trains the machine learning ModelsProduction using the provided training DataTrain and saves the trained ModelsProduction to disk.

        eval(ModelsProduction, x, y) -> Dict[str, float]:
            Evaluates the trained ModelsProduction on the given dataset and computes various performance MetricsTrain.

        run():
            Executes the complete pipeline, including scaling selection, training, evaluation,
            and logging of parameters and MetricsTrain via MLFlow.

    Usage:
        Instantiate the `TrainerPipeline` class and call the `run` method to execute the pipeline.

    Raises:
        KeyError: If the specified configuration section is missing or incorrect.
        pydantic.ValidationError: If configuration validation fails.
        FileNotFoundError: If dataset files are missing.
        ValueError: If an invalid scaling option is specified or ModelsProduction compatibility issues arise.
    """

    def __init__(
        self,
        config_model: PydanticConfigModel,
        config_loader: HydraConfLoader,
        experiment_tracker: MLFlowTracker,
        config_name: str,
        config_section: str,
        model: BaseEstimator,
    ):
        """
        Initializes the TrainerPipeline with configuration, experiment tracking, and ModelsProduction.

        Parameters:
            config_model (PydanticConfigModel): The configuration ModelsProduction for validation.
            config_loader (HydraConfLoader): The Hydra configuration loader for handling YAML files.
            experiment_tracker (MLFlowTracker): An instance for tracking ML experiments.
            config_name (str): The name of the configuration file. Defaults to "config".
            config_section (str): The configuration section to load. Defaults to "train_config".
            model (BaseEstimator): A scikit-learn ModelsProduction to train. Defaults to `SVC()`.

        Raises:
            pydantic.ValidationError: If the configuration validation fails.
        """

        super().__init__(
            config_model=config_model,
            config_loader=config_loader,
            experiment_tracker=experiment_tracker,
            config_name=config_name,
            config_section=config_section,
            model=model,
        )
        self.normalized_df = self.valid_config.normalized_df
        self.standardized_df = self.valid_config.standardized_df

    def scaling_selection(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the type of DataTrain scaling to apply based on configuration.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the training and testing sets:
            (x_train, x_test, y_train, y_test).

        Raises:
            FileNotFoundError: If the specified dataset files are not found.
            ValueError: If no scaling option is specified.
        """
        if self.normalized_df:
            x_train = (
                pl.scan_csv(self.valid_config.normalized_x_train).collect().to_numpy()
            )
            y_train = (
                pl.scan_csv(self.valid_config.y_train).collect().to_numpy().ravel()
            )
            x_test = (
                pl.scan_csv(self.valid_config.normalized_x_test).collect().to_numpy()
            )
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy().ravel()
        elif self.standardized_df:
            x_train = (
                pl.scan_csv(self.valid_config.standardized_x_train).collect().to_numpy()
            )
            y_train = (
                pl.scan_csv(self.valid_config.y_train).collect().to_numpy().ravel()
            )
            x_test = (
                pl.scan_csv(self.valid_config.standardized_x_test).collect().to_numpy()
            )
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy().ravel()
        else:
            x_train = pl.scan_csv(self.valid_config.x_train).collect().to_numpy()
            y_train = (
                pl.scan_csv(self.valid_config.y_train).collect().to_numpy().ravel()
            )
            x_test = pl.scan_csv(self.valid_config.x_test).collect().to_numpy()
            y_test = pl.scan_csv(self.valid_config.y_test).collect().to_numpy().ravel()

        return x_train, x_test, y_train, y_test

    def train(self, x: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Trains the machine learning ModelsProduction with the provided DataTrain and saves the trained ModelsProduction.

        Parameters:
            x (np.ndarray): The training input DataTrain.
            y (np.ndarray): The training target labels.

        Returns:
            BaseEstimator: The trained scikit-learn ModelsProduction.

        Raises:
            json.JSONDecodeError: If the parameter configuration file cannot be loaded.
            IOError: If the ModelsProduction file cannot be saved.
        """
        if self.valid_config.using_custom_parameters:
            logging.info("Using custom parameters to train the ModelsProduction.")
            model_ = self.model.set_params(**self.valid_config.custom_parameters)
            model_.fit(x, y)

        else:
            logging.info("Using tuned parameters with CV")
            with open(self.valid_config.tuned_parameters) as f:
                print(self.valid_config.tuned_parameters)
                parameters = json.load(f)
                f.close()
            model_ = self.model.set_params(**parameters)
            model_.fit(x, y)

        joblib.dump(
            model_,
            f"{self.valid_config.model_path}/{self.valid_config.experiment_name}.joblib",
        )
        return model_

    @override
    def eval(self, x, y, model: BaseEstimator = None) -> Tuple[Dict[str, float], Any]:
        """
        Evaluates the ModelsProduction's performance on the given dataset and computes evaluation MetricsTrain.

        Parameters:
            model (BaseEstimator): The trained scikit-learn ModelsProduction to evaluate.
            x (np.ndarray): The input DataTrain for evaluation.
            y (np.ndarray): The true labels for evaluation.

        Returns:
            dict: A dictionary containing the computed evaluation MetricsTrain:
                - roc_auc
                - average_precision
                - accuracy
                - precision
                - recall
                - f1

        Raises:
            ValueError: If the ModelsProduction does not support `predict_proba` for probability prediction.
        """
        y_predicted = model.predict(x)  # type: ignore
        mlflow_signature = infer_signature(x, y_predicted)
        params_dict = {
            "accuracy": accuracy_score(y, y_predicted),
            "precision": precision_score(
                y, y_predicted, average=self.valid_config.average_mode
            ).item(),
            "recall": recall_score(
                y, y_predicted, average=self.valid_config.average_mode
            ).item(),
            "f1": f1_score(
                y, y_predicted, average=self.valid_config.average_mode
            ).item(),
        }

        # Check if the ModelsProduction supports `predict_proba`
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x)
            params_dict["roc_auc"] = roc_auc_score(y, y_proba, multi_class="ovr")
            params_dict["average_precision"] = average_precision_score(y, y_proba)

        return params_dict, mlflow_signature

    def run(self):
        """ "
        Executes the complete training pipeline, including scaling, training, evaluation,
        and tracking parameters and MetricsTrain with MLFlow.

        Steps:
            1. Loads the training and testing datasets using the configured scaling option.
            2. Starts an MLFlow experiment run.
            3. Trains the ModelsProduction with the training dataset and logs parameters.
            4. Evaluates the ModelsProduction on both training and testing datasets.
            5. Logs evaluation MetricsTrain and saves them to disk.

        Raises:
            Exception: Logs and raises any exception encountered during pipeline execution.
        """
        try:
            logging.info("Staring the Trainer...")
            x_train, x_test, y_train, y_test = self.scaling_selection()
            exp_id = self.experiment_tracker.get_or_create_experiment_id(
                name=self.valid_config.experiment_name
            )
            logging.info("Test and train sets loaded and experiment ID created")
            logging.info("Starting the tracking with MLFlow...")
            with self.experiment_tracker.initialize_experiment(experiment_id=exp_id):
                logging.info("Training ModelsProduction...")
                ml_model = self.train(x_train, y_train)
                model_class = {"model__class": type(ml_model).__name__}
                model_parameters = {
                    f"{self.valid_config.experiment_name}__{k}": v
                    for k, v in ml_model.get_params().items()
                }
                model_and_parameters = model_class | model_parameters
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_parameters.json",
                    "w",
                ) as f:
                    json.dump(model_and_parameters, f)  # type: ignore
                self.experiment_tracker.log_param(
                    key="model__class", value=type(ml_model).__name__
                )
                self.experiment_tracker.log_params(model_parameters)
                logging.info(f"The parameters saved are:\n{model_and_parameters}")
                logging.info("Model trained and parameters saved.")

                logging.info("Evaluating ModelsProduction..")
                train_metrics, _ = self.eval(model=ml_model, x=x_train, y=y_train)
                train_metrics_to_save = {
                    f"train__{k}": v for k, v in train_metrics.items()
                }
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_Train_metrics.json",
                    "w",
                ) as f:
                    json.dump(train_metrics_to_save, f)  # type: ignore

                self.experiment_tracker.log_metrics(train_metrics_to_save)

                test_metrics, mlflow_signature = self.eval(
                    model=ml_model, x=x_test, y=y_test
                )
                test_metrics_to_save = {
                    f"test__{k}": v for k, v in test_metrics.items()
                }
                with open(
                    self.valid_config.metrics
                    / f"{self.valid_config.experiment_name}_Test_metrics.json",
                    "w",
                ) as f:
                    json.dump(test_metrics_to_save, f)  # type: ignore
                self.experiment_tracker.log_metrics(test_metrics_to_save)
                logging.info(
                    "Model Evaluated and train and test MetricsTrain tracked and saved"
                )
                self.experiment_tracker.log_model_signature(
                    model=ml_model,
                    signature=mlflow_signature,
                    registered_model_name=self.valid_config.registered_model_name,
                )
                logging.info("train and test completed.")
        except Exception as e:
            logging.error(f"Trainer failure at {e}")
            raise e
