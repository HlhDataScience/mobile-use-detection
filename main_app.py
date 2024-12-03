"""The main API app program entry point"""

# Load the classifier
from production_phase.src.api.api import setup_app
from production_phase.src.api.Frameworks import FastAPIFramework
from production_phase.src.predictions.predict import load_classifier

# CONSTANST:
MODEL_PATH = "production_phase/model/Tree_Classifier_GridSearchCV.pkl"

classifier = load_classifier(model_path=MODEL_PATH)
framework = FastAPIFramework()

if __name__ == "__main__":
    setup_app(framework, classifier)
    framework.run()
