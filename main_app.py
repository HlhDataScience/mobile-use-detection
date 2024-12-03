"""The main API app program entry point"""

from crypt import methods

# Load the classifier
from production_phase.src.api.api import read_item, root, setup_app
from production_phase.src.api.Frameworks import FastAPIFramework
from production_phase.src.predictions.predict import load_classifier

# CONSTANST:
MODEL_PATH = "EDA_train_phase/models/Tree_Classifier_New_v4.joblib"

classifier = load_classifier(model_path=MODEL_PATH)
framework = FastAPIFramework()
framework.add_route(path="/", endpoint=root, methods=["GET"])
framework.add_route(path="/items/{item_id}", endpoint=read_item, methods=["GET"])
setup_app(framework, classifier)
app = framework.app
if __name__ == "__main__":
    framework.run(port=8001)
