"""main entry point of the program"""
from EDA_train_phase.src.logging_functions.logger import setup_logging 
from EDA_train_phase.src.pipeline.transformation_pipeline import LazyTransformationPipeline
from EDA_train_phase.src.training.train import TrainerPipeline
from PathLib import Path

#CONSTANTS

LOG_FILE = Path("logs/main_program.log")
setup_logging(LOG_FILE)

def main()-> None:
    
    logging.info("Initializing Program...")
                 
    pipeline = LazyTransformationPipeline()
    train = TrainerPipeline()
    
    logging.info("Data Transformation and Trainer instanciated")
    logging.info("Running Pipelines")
    
    try:
        pipeline.run()
        train.run()
        
    except Exception as e:
        logging.error(f"An error found at {e}")

if __name__ == '__main__':
    main()
    
