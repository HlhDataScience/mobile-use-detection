"""main entry point of the program"""

import argparse
import logging
from pathlib import Path

import dagshub
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

from EDA_train_phase.src.logging_functions.logger import setup_logging
from EDA_train_phase.src.pipeline.transformation_pipeline import (
    LazyTransformationPipeline,
)
from EDA_train_phase.src.training.train import TrainerPipeline

# CONSTANTS
LOG_FILE = Path("logs/main_program.log")
CONFIG_PATH = "conf/"
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


# Main program
def main(args) -> None:
    """Main Program function that handles all the pipelines and tracks the experiment using MLFlow and dagshub."""
    logging.info("Initializing Program...")

    pipeline = LazyTransformationPipeline()
    train = TrainerPipeline()

    logging.info("Data Transformation and Trainer instantiated")
    logging.info("Running Pipelines")

    try:
        if args.pipeline in ["all", "transformation"]:
            pipeline.run()
            logging.info("Transformation pipeline completed.")

        if args.pipeline in ["all", "training"]:
            train.run()
            logging.info("Training pipeline completed.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


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
