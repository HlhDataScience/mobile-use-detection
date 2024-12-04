"""The main API _app program entry point"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

# Load the classifier
from production_phase.src.api.apifuncs import (
    classify,
    read_item,
    root,
    setup_app,
)
from production_phase.src.api.Frameworks import FastAPIFramework

# CONSTANTS:
API_CONSTRUCTOR = {
    "/": (root, ["GET"]),
    "/predict/": (classify, ["POST"]),
    "/items/{item_id}": (read_item, ["GET"]),
}


framework = setup_app(FastAPIFramework(app=FastAPI()), api_functions=API_CONSTRUCTOR)
app = framework.app  # expose the _app to make function the fastapi CLI
if __name__ == "__main__":
    framework.run(port=8001)
