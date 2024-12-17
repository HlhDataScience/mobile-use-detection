"""The main API and app program entry point"""

from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

from apifuncs import classify, get_results, load_classifier, read_root, results
from fastapi import FastAPI
from Frameworks import FastAPIFramework
from pydantic import BaseModel
from validation_classes import APIInfo, ClassifierOutput, ResultsDisplay
from WebFrameworksProtocols import EndPointProtocolFunction

# CONSTANTS
MODEL_PATH = "Tree_Classifier_New_v4.joblib"
# noinspection PyTypeChecker
API_CONSTRUCTOR: Dict[str, Tuple[EndPointProtocolFunction, List[str], BaseModel]] = {
    "/": (read_root, ["GET"], APIInfo),
    "/predict/": (classify, ["POST"], ClassifierOutput),
    "/results/": (results, ["GET"], ResultsDisplay),
    "/results/get_results/": (get_results, ["GET"], ResultsDisplay),
}


# Launch API
# noinspection PyShadowingNames
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager to handle the lifecycle of the FastAPI app.

    This function loads the classifier model on startup and ensures it is
    unloaded when the app shuts down.

    Args:
        app (FastAPI): The FastAPI app instance.

    Yields:
        None
    """
    try:
        classifier = load_classifier(model_path=MODEL_PATH)
        app.state.classifier = classifier
        yield
    except Exception as e:
        print(f"Error during classifier loading: {e}")
        yield
    finally:
        if hasattr(app.state, "classifier"):
            del app.state.classifier


framework = FastAPIFramework.from_constructor(
    app_instance=FastAPI(lifespan=lifespan), api_functions=API_CONSTRUCTOR
)
app = framework.app
