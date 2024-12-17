# Mobile Usage Detection

## Overview

This repository demonstrates the implementation of efficient MLOps workflows using accessible tools to achieve effective results. We utilize GitHub and DVC for version control of both code and data, ensuring consistency and traceability. Pydantic is employed to enforce data input serialization during Exploratory Data Analysis (EDA) and training phases, while FastAPI serves as the backend framework. The project features a Gradio-powered application and a Docker image for seamless deployment in cloud environments. To enhance code reusability, we implement interfaces and abstract classes using Python's `protocol` and `abc` modules, promoting portability and adaptability.

## Project Structure

The project is organized into two primary directories:

- **`images/`**: Contains Docker-ready files. It employs the interface strategy pattern to avoid reliance on a specific framework. For testing purposes, FastAPI and Streamlit were selected due to their performance and compatibility.

- **`src/`**: Houses modular source code responsible for Exploratory Data Analysis (EDA), data transformation pipelines, and the TrainerPipeline. Abstractions are utilized to ensure modules are reusable across different frameworks.

## Usage Instructions

The repository is designed to be used via the Docker image; however, the source code is also provided for customization. Two primary methods are available for utilizing this repository:

1. **Using the Training Pipeline via Makefile**:

```bash
git clone https://github.com/HlhDataScience/mobile-use-detection.git
cd mobile-use-detection
make run_all

```
**Using the Docker Compose**
```
cd images
docker-compose up --build
```
## Done related to EDA_train_phase
- We have completed the Hydra config to handle yalm automatization of the pipeline.
- We have created the `logger.py` file that handles logging across the files.
- We have completed the data validation & serialization classes
- We have performed the EDA analysis
- Created an `ABC Pipeline` class to be aligned with SOLID principles.
- Created `ABC validation` classes and its interfaces for data and configuration validation
- Tested `LazyTransformationPipeline` refactored as a subclass of `BasicPipeline(ABC)` abstract class.
- Tested the run method of the pipeline with interfaces and good results.
- Tested the rest of the methods of CV with good results.
- Completed `train.py` class.
- Completed test for `TrainerPipeline`: Everything works properly.
- Created `Makefile` for `main.py`
- Tested `main.py`. It works without problem.
- Created `KNN` deviating branch to test different experiments.
- Created the `dvc` pipeline managing system
- Created class methods for `ABC_Pipeline`. `ABC_Pipeline` now is more flexible.

## Done related to production_phase
  - Refactored structure of the project to work with the 2 development and production phases of MLOps
    - Created the FastAPI app
  - Created the agnostic processor of the API framework
  - System of dev fastapi works with agnostic implementation
  - Infrastructure for the web app created.
  - Created class methods and protocols for the api functions
  - Included headers and root information
  - Included pydantic validation through the predict, results and get_results methods
  - Included query functionality for the small json file used as database.
  - All the functions of the API adhere to EndPointProtocol class.
  - Created the interfaces for the app and tested them in jupyter notebook
  - Updated Makefile to support the different apps.
  - Completed refactor after dockerization of the project.
    
## TODOs related to EDA_train_phase
  - ~~We need to create Hydra Config to handle yalm automatization~~ 
  - ~~We need to create a logging function to handle the whole program~~
  - ~~We need to create the data and config validation classes~~
  - ~~We need to perform EDA analysis~~
  - ~~We need to create an ABC Pipeline class to adhere to SOLID principles~~
  - ~~We need to create an ABC validation classes and its interfaces~~
  - ~~We need to refactor the TransformationPipeline to MAke it inherit from ABC_Pipeline.py~~
  - ~~We need to test the TransformationPipeline and its different methods~~
  - ~~We need to test the other CV methods in the TransformationPipeline~~
  - ~~We need to complete the train.py.~~
  - ~~We need to test the TrainerPipeline to check everything is working properly.~~
  - ~~We need to create a makefile for the main.py~~
  - ~~We need to use the main.py to run the whole program~~
  - ~~We need to create a deviating branch(or several) to test other classic machine learning models.~~
  - ~~We need to create a dvc pipeline managing system~~
  - ~~We need to create class methods and typing adherence for the functions~~
  - We need to create test for every class and method with pytest

## TODOs related to production_phase
 - ~~Completed refactor of the project's structure.~~
 -  ~~Create the FASTAPI app~~
 - ~~Create an agnostic processor of webapp and webframeworks using protocols.~~
 - ~~make the system of fastapi dev works with agnostic interface.~~
 - ~~create the infrastructure of the API for the web app~~
 - ~~Create class methods and protocols for the api functions.~~
 - ~~Include Headers and root information~~
 - ~~Improve FastApi functions with pydantic models~~
 - ~~Include query functionality for the small json file.~~
 - ~~Check with dir() all the functions that need to adhere to EndPointProtocolFunction~~
 - ~~Create the interface system for the webapp and test it with jupỳter notebook.~~
 - ~~Update the Makefile to support the different apps~~
 - ~~Dockerize the project to make it portable~~
