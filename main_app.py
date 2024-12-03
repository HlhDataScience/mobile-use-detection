"""The main API app program entry point"""

from crypt import methods

# Load the classifier
from production_phase.src.api.apifuncs import (
    predict_endpoint,
    read_item,
    root,
    setup_app,
)
from production_phase.src.api.Frameworks import FastAPIFramework
from production_phase.src.predictions.predict import load_classifier

# CONSTANT:
MODEL_PATH = "EDA_train_phase/models/Tree_Classifier_New_v4.joblib"
api_dict = {
    "/": (root, ["GET"]),
    "/predict/": (predict_endpoint, ["GET"]),
    "/items/{item_id}": (read_item, ["GET"]),
}
classifier = load_classifier(model_path=MODEL_PATH)

framework = setup_app(FastAPIFramework(), api_functions=api_dict)
app = framework.app
if __name__ == "__main__":
    framework.run(port=8001)
