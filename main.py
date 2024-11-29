"""main entry point of the program"""

import argparse
import logging
from pathlib import Path

import dagshub
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from sklearn.svm import SVC

from EDA_train_phase.src.logging_functions.logger import setup_logging
from EDA_train_phase.src.pipeline.transformation_pipeline import (
    LazyTransformationPipeline,
)
from EDA_train_phase.src.training.train import TrainerPipeline
from EDA_train_phase.src.validation_classes.validation_configurations import (
    DataTransformationConfig,
    DataValidationConfig,
    TrainerConfig,
)
from EDA_train_phase.src.validation_classes.validation_interfaces import (
    HydraConfLoader,
    MLFlowTracker,
    PanderaValidationModel,
    PydanticConfigModel,
)

# CONSTANTS
LOG_FILE = Path("logs/main_program.log")
CONFIG_PATH = "EDA_train_phase/conf/"
DAGSHUB_REPO_OWNER = "data_analitics_HLH"
DAGSHUB_REPO = "mobile-use-detection"

# Logging and configuration
setup_logging(LOG_FILE)
logging.info("Setting up Hydra")
if GlobalHydra().is_initialized():
    GlobalHydra.instance().clear()
initialize(config_path=CONFIG_PATH, job_name="load_config")
logging.info("Completed")
logging.info("Setting up DagsHUb with MLFlow tracking")
dagshub.init(DAGSHUB_REPO, DAGSHUB_REPO_OWNER, mlflow=True)
logging.info("Completed. Ready for main program.")

# Loading interfaces and validations
vali_model = PanderaValidationModel(validation_model=DataValidationConfig)
confi_model_data = PydanticConfigModel(config_model=DataTransformationConfig)
confi_model_trainer = PydanticConfigModel(config_model=TrainerConfig)
hydra_loader_conf = HydraConfLoader()
exp_tracker = MLFlowTracker()


# Main program
def main(args) -> None:
    """Main Program function that handles all the pipelines and tracks the experiment using MLFlow and dagshub."""
    logging.info("Initializing Program...")

    transformation_pipeline = LazyTransformationPipeline(
        validation_model=vali_model,
        config_model=confi_model_data,
        config_loader=hydra_loader_conf,
        config_name="config",
        config_section="transformation_config",
        apply_custom_function=False,
        model=SVC(),
    )
    logging.info("Data Transformation instantiated")

    logging.info("Pipelines running")
    try:
        if args.pipeline in ["all", "transformation"]:
            transformation_pipeline.run()
            logging.info("Transformation pipeline completed.")

        # Instantiate the TrainerPipeline only after the transformation pipeline has run successfully
        if args.pipeline in ["all", "training"]:
            try:
                train_pipeline = TrainerPipeline(
                    config_model=confi_model_trainer,
                    config_loader=hydra_loader_conf,
                    experiment_tracker=exp_tracker,
                    config_name="config",
                    config_section="train_config",
                    model=SVC(),
                )
                logging.info("Trainer instantiated")
                train_pipeline.run()
                logging.info("Training pipeline completed.")
            except Exception as e:
                logging.error(f"Failed instantiating Trainer: {e}")

    except Exception as e:
        logging.exception(f"An error occurred during pipeline execution:\n{e}.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Run data transformation and training pipelines."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "transformation", "training"],
        default="all",
        help="Specify which pipeline to run: 'all', 'transformation', or 'training'. Default is 'all'.",
    )
    args_ = parser.parse_args()

    main(args_)
