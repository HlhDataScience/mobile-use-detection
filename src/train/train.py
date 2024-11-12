import dagshub
import mlflow
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

DAGSHUB_REPO_OWNER = "<username>"
DAGSHUB_REPO = "DAGsHub-Tutorial"
dagshub.init(DAGSHUB_REPO, DAGSHUB_REPO_OWNER)

# Consts
CLASS_LABEL = "MachineLearning"
train_df_path = "data/train.csv"
test_df_path = "data/test.csv"


def get_or_create_experiment_id(name):
    """This function handle the creation of the mlflow experiment id tracking"""
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def feature_engineering(raw_df):
    """Develops the feature engineering function"""
    df = raw_df.copy()
    df["CreationDate"] = pd.to_datetime(df["CreationDate"])
    df["CreationDate_Epoch"] = df["CreationDate"].astype("int64") // 10**9
    df = df.drop(columns=["Id", "Tags"])
    df["Title_Len"] = df.Title.str.len()
    df["Body_Len"] = df.Body.str.len()
    # Drop the correlated features
    df = df.drop(columns=["FavoriteCount"])
    df["Text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")
    return df


def eval_model(clf, x, y):
    """evaluates the model"""
    y_proba = clf.predict_proba(x)[:, 1]
    y_pred = clf.predict(x)
    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def split(random_state=42):
    """splits the data into the train and test sets"""
    print("Loading data...")
    df = pd.read_csv("data/CrossValidated-Questions.csv")
    df[CLASS_LABEL] = df["Tags"].str.contains("machine-learning").fillna(False)
    train_df, test_df = train_test_split(
        df, random_state=random_state, stratify=df[CLASS_LABEL]
    )

    print("Saving split data...")
    train_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)
