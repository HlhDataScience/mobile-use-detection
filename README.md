# Mobile Usage Detection

## About

This repository is a small sample of how we can leverage MLOps workflow with simple tools and good results. We use github and dvc to version control both our code and our data. We also use pydantic to ensure good data input serialization (for EDA and training steps) as well as backend with FastAPI. Finally we have a gradio powered app and a Docker image to use it in cloud computing.

## Done
  - We have performed the EDA analysis
  - We have completed the data validation & serialization classes
  - We have completed the optim search models within the LazyTransformationPipeline
  - We have changed the FilePath to take advantage of pathlib
  - We have created the logger.py file that handles logging across the files.
  - We have completed the Hydra config to handle yalm automatization o the pipeline.

## TODOs related to train section
  - We need to test all the classes developed on transformation_pipeline within the jupyter notebook. 
  - To improve EDA wwe will need to transform all the functions into a class and test it into the EDA notebook.
  - We need to complete the train.py and generate the Makefiles to automate the process.
  - We need to create a deviating branch(or several) to test other classic machine learning models.
  - Suggested test duplicating the data until we reach 3 million rows. We can use this as stress test for speed in the data pipeline


## toDOs related to API, app and dockerization 
