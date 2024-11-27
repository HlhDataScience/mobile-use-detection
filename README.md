# Mobile Usage Detection

## About

This repository is a small sample of how we can leverage MLOps workflow with simple tools and good results. We use GitHub and dvc to version control both our code and our data. We also use pydantic to ensure good data input serialization (for EDA and training steps) as well as backend with FastAPI. Finally, we have a gradio powered app and a Docker image to use it in cloud computing.

**The project is divided into 2 directories**:
  - EDA_train_phase : IN this directory you will find the full trnasformation and training pipelines. Based on ABC, the code uses interfaces and abstractions be reusable in other projects with different needs.
  - Production_phase: This directory contains the basic implementation of the app using FastAPI and Gradio to dockerization. In this way can be portable into other systems (local, owned server or cloud computing)

## Done related to EDA_train_phase
  - Tested the rest of the methods of CV with good results.
  - Tested run method of the pipeline with interfaces and good results.
  - Tested LazyTransformationPipeline refactored as subclass of BasicPipeline(ABC) abstract class.
  - Created ABC validation classes and its interfaces for data and configuration validation
  - Created and ABC Pipeline class to be aligned with SOLID principles.
  - Refactored the whole directory to work with Hydra general config
  - We have performed the EDA analysis
  - We have completed the data validation & serialization classes
  - We have completed the optim search models within the LazyTransformationPipeline
  - We have changed the FilePath to take advantage of pathlib
  - We have created the logger.py file that handles logging across the files.
  - We have completed the Hydra config to handle yalm automatization of the pipeline.
  - Test 1 completed: Validation and configuration works within the LazyTransformationPipeline.
  - Implemented basic EDA class to be tested later


## TODOs related to train section
  - We need to complete the train.py and generate the Makefiles to automate the process.
  - We need to create a deviating branch(or several) to test other classic machine learning models.
  - Suggested test duplicating the data until we reach 3 million rows. We can use this as stress test for speed in the data pipeline


## Done related to production_phase
  - Refactored structure of the project to work with the 2 development and production phases of MLOps

## toDOs related to production_phase

