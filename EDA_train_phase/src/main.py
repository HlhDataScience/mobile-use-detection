"""main entry point of the program"""

import logging
from pathlib import Path

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
def main() -> None:
    """Main Program function that handles all the pipelines adn tracks the experiment using MLFlow and dagshub."""

    logging.info("Initializing Program...")

    pipeline = LazyTransformationPipeline()
    train = TrainerPipeline()

    logging.info("Data Transformation and Trainer instantiated")
    logging.info("Running Pipelines")

    try:
        pipeline.run()
        train.run()

    except Exception as e:
        logging.error(f"An error found at {e}")


if __name__ == "__main__":
    main()
