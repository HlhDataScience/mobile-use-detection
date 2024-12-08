"""The main API _app program entry point"""

from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

from Phases.SrcProduction.api.apifuncs import (
    classify,
    get_results,
    load_classifier,
    read_root,
    results,
)
from Phases.SrcProduction.api.Frameworks import FastAPIFramework
from Phases.SrcProduction.api.validation_classes import (
    APIInfo,
    ClassifierOutput,
    ResultsDisplay,
)
from Phases.SrcProduction.interfaces.WebFrameworksProtocols import (
    EndPointProtocolFunction,
)

# CONSTANTS:
MODEL_PATH = "Phases/ModelsProduction/Tree_Classifier_New_v4.joblib"
# noinspection PyTypeChecker
API_CONSTRUCTOR: Dict[str, Tuple[EndPointProtocolFunction, List[str], BaseModel]] = {
    "/": (read_root, ["GET"], APIInfo),
    "/predict/": (classify, ["POST"], ClassifierOutput),
    "/results/": (results, ["GET"], ResultsDisplay),
    "/results/get_results/": (get_results, ["GET"], ResultsDisplay),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """controller to load and delete the classifier."""
    classifier = load_classifier(model_path=MODEL_PATH)
    app.state.classifier = classifier
    yield
    del app.state.classifier


framework = FastAPIFramework.from_constructor(
    app_instance=FastAPI(lifespan=lifespan), api_functions=API_CONSTRUCTOR
)

app = framework.app  # expose the _app to make function the fastapi CLI

if __name__ == "__main__":
    framework.run(port=8001)
