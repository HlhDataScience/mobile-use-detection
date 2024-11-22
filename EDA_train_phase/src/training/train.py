""" THis module will be used to perform the train_data test_data pipeline"""

import dagshub
import mlflow
from transformation_pipeline import LazyTransformationPipeline

""" THis module will be used to perform the train_data test_data pipeline"""

DAGSHUB_REPO_OWNER = "<username>"
DAGSHUB_REPO = "DAGsHub-Tutorial"
dagshub.init(DAGSHUB_REPO, DAGSHUB_REPO_OWNER)

# Consts
CLASS_LABEL = "MachineLearning"
train_df_path = "data/train_data.csv"
test_df_path = "data/test_data.csv"


def get_or_create_experiment_id(name):
    """This function handle the creation of the mlflow experiment id tracking"""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id
