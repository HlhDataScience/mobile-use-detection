"""Module for the Validation Protocols interfaces"""

from collections.abc import Coroutine
from typing import Any, Dict, List, Sequence

from fastapi import FastAPI

from production_phase.src.interfaces.WebFrameworksProtocols import (
    WebFrameworkProtocol,
)


class FastAPIFramework(WebFrameworkProtocol):  # Inherit from the protocol
    """Interface class that implements the protocols"""

    def __init__(self, app: FastAPI):
        self.app = app

    def add_route(
        self, path: str, endpoint: Coroutine[Any, Any, Dict], methods: Sequence[str]
    ) -> None:
        """agnostic get and post methods for routing"""
        for method in methods:
            if method.lower() == "get":
                self.app.get(path)(endpoint)
            elif method.lower() == "post":
                self.app.post(path)(endpoint)

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """specific server running"""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
