"""main entry point of the program"""

import logging
from pathlib import Path
import argparse

import dagshub

from EDA_train_phase.src.logging_functions.logger import setup_logging
from EDA_train_phase.src.pipeline.transformation_pipeline import (
    LazyTransformationPipeline,
)
from EDA_train_phase.src.training.train import TrainerPipeline

# CONSTANTS
DAGSHUB_REPO_OWNER = "<username>"
DAGSHUB_REPO = "DAGsHub-Tutorial"
dagshub.init(DAGSHUB_REPO, DAGSHUB_REPO_OWNER)
LOG_FILE = Path("logs/main_program.log")
setup_logging(LOG_FILE)


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
    parser = argparse.ArgumentParser(description="Run data transformation and training pipelines.")
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["all", "transformation", "training"],
        default="all",
        help="Specify which pipeline to run: 'all', 'transformation', or 'training'. Default is 'all'.",
    )
    args = parser.parse_args()

    main(args)
