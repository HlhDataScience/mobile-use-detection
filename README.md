# Mobile Usage Detection

## About

This repository is a small sample of how we can leverage MLOps workflow with simple tools and good results. We use GitHub and dvc to version control both our code and our data. We also use pydantic to ensure good data input serialization (for EDA and training steps) as well as backend with FastAPI. Finally, we have a gradio powered app and a Docker image to use it in cloud computing.

**The project is divided into 2 directories**:
  - EDA_train_phase : IN this directory you will find the full transformation and training pipelines. Based on ABC, the code uses interfaces and abstractions be reusable in other projects with different needs.
  - Production_phase: This directory contains the basic implementation of the app using FastAPI and Gradio to dockerization. In this way can be portable into other systems (local, owned server or cloud computing)

## Done related to EDA_train_phase
- We have completed the Hydra config to handle yalm automatization of the pipeline.
- We have created the logger.py file that handles logging across the files.
- We have completed the data validation & serialization classes
- We have performed the EDA analysis
- Created an ABC Pipeline class to be aligned with SOLID principles.
- Created ABC validation classes and its interfaces for data and configuration validation
- Tested LazyTransformationPipeline refactored as a subclass of BasicPipeline(ABC) abstract class.
- Tested the run method of the pipeline with interfaces and good results.
- Tested the rest of the methods of CV with good results.
- Completed train.py class.
- Completed test for TrainerPipeline: Everything works properly.


## TODOs related to EDA_train_phase
  - ~~We need to create Hydra Config to handle yalm automatization~~ 
  - ~~We need to create a logging function to handle the whole program~~
  - ~~We need to create the data and config validation classes~~
  - ~~We need to perform EDA analysis~~
  - ~~We need to create a ABC Pipeline class to adhere to SOLID principles~~
  - ~~We need to create a ABC validation classes and its interfaces~~
  - ~~We need to refactor the TransformationPipeline to MAke it inherit from ABC_Pipeline.py~~
  - ~~We need to test the TransformationPipeline and its different methods~~
  - ~~We need to test the other CV methods in the TransformationPipeline~~
  - ~~We need to complete the train.py.~~
  - ~~We need to test the TrainerPipeline to check everything is working properly.~~
  - We need to use the main.py to run the whole program
  - We need to create a deviating branch(or several) to test other classic machine learning models.
  - Suggested test duplicating the data until we reach 3 million rows. We can use this as stress test for speed in the data pipeline


## Done related to production_phase
  - Refactored structure of the project to work with the 2 development and production phases of MLOps

## toDOs related to production_phase
 - Create the FASTAPI app
 - ~~Completed refactor of the project's structure.~~
