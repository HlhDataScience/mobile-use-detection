"""Module for the Validation Protocols interfaces"""

from collections.abc import Coroutine
from typing import Any, Dict, Sequence

from fastapi import FastAPI


class FastAPIFramework:  # Inherit from the protocol
    """Interface class that implements the protocols"""

    def __init__(self, app: FastAPI):
        self.app = app

    def add_route(
        self, path: str, endpoint: Coroutine[Any, Any, Dict], methods: Sequence[str]
    ) -> None:
        """agnostic get and post methods for routing"""
        for method in methods:
            if method.lower() == "get":
                self.app.get(path)(endpoint)  # type: ignore
            elif method.lower() == "post":
                self.app.post(path)(endpoint)  # type: ignore

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """specific server running"""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
