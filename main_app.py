"""The main API _app program entry point"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from EDA_train_phase.src.logging_functions.logger import setup_logging
from production_phase.src.api.apifuncs import (
    classify,
    get_results,
    load_classifier,
    results,
    root,
    setup_app,
)
from production_phase.src.api.Frameworks import FastAPIFramework

# CONSTANTS:
MODEL_PATH = "EDA_train_phase/models/Tree_Classifier_New_v4.joblib"
API_CONSTRUCTOR = {
    "/": (root, ["GET"]),
    "/predict/": (classify, ["POST"]),
    "/results/": (results, ["GET"]),
    "/results/{index_id}": (get_results, ["GET"]),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """controller to load and delete the classifier."""
    classifier = load_classifier(model_path=MODEL_PATH)
    app.state.classifier = classifier
    yield
    del app.state.classifier


framework = FastAPIFramework(app=FastAPI(lifespan=lifespan))
framework = setup_app(framework=framework, api_functions=API_CONSTRUCTOR)  # type: ignore
app = framework.app  # expose the _app to make function the fastapi CLI

if __name__ == "__main__":
    framework.run(port=8001)
