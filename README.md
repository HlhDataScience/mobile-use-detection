# Mobile Usage Detection

## About

This repository is a small sample of how we can leverage MLOps workflow with simple tools and good results. We use github and dvc to version control both our code and our data. We also use pydantic to ensure good data input serialization (for EDA and training steps) as well as backend with FastAPI. Finally we have a gradio powered app and a Docker image to use it in cloud computing.

## Done
  - We have performed the EDA analysis

## toDOs related to train section
  - We need to complete the class for data validation & serialization using pandera, polars & pydantic
  - We need to complete the optim search models within LazyTransformationPipeline class & test them in a jupyter notebook.
  - We need to complete the train.py and generate the Makefiles to automate the process.
  - We need to create a devation branch(or several) to test orther classic machine learning models.

## toDOs related to API, app and dockerization 
