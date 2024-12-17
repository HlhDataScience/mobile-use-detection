"""Module for the Validation Protocols interfaces"""

from typing import Dict, List, Optional, Sequence, Tuple

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from src.production.backend.WebFrameworksProtocols import (
    EndPointProtocolFunction,
    WebFrameworkProtocol,
)


class FastAPIFramework:
    """
    A framework interface class implementing the WebFrameworkProtocol.

    This class integrates the FastAPI web framework, allowing for the addition
    of routes and server execution, adhering to a unified protocol definition.
    """

    def __init__(self, app: FastAPI):
        """
        Initializes the FastAPIFramework with a FastAPI app instance.

        Args:
            app (FastAPI): An instance of the FastAPI application.
        """
        self.app = app

    def add_route(
        self,
        path: str,
        endpoint: EndPointProtocolFunction,
        methods: Sequence[str],
        response_model: Optional[BaseModel] = None,
        status_code: Optional[int] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[Depends]] = None,
    ) -> None:
        """
        Adds routes to the FastAPI app for the specified HTTP methods.

        Args:
            path (str): The route path.
            endpoint (EndPointProtocolFunction): The endpoint function to handle requests.
            methods (Sequence[str]): A sequence of HTTP methods (e.g., ['GET', 'POST']).
            response_model (Optional[type]): The response model for the route.
            status_code (Optional[int]): The HTTP status code for the response.
            tags (Optional[List[str]]): Tags for API documentation grouping.
            dependencies (Optional[List[Depends]]): Dependencies to inject for the endpoint.

        Raises:
            ValueError: If an unsupported HTTP method is provided.
        """
        for method in methods:
            if method.lower() == "get":
                self.app.get(
                    path,
                    response_model=response_model,
                    status_code=status_code,
                    tags=tags,
                    dependencies=dependencies,
                )(
                    endpoint
                )  # type: ignore
            elif method.lower() == "post":
                self.app.post(
                    path,
                    response_model=response_model,
                    status_code=status_code,
                    tags=tags,
                    dependencies=dependencies,
                )(
                    endpoint
                )  # type: ignore
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Starts the FastAPI app using Uvicorn.

        Args:
            host (str): The host address to bind the server to. Default is "127.0.0.1".
            port (int): The port number to run the server on. Default is 8000.
        """
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)

    @classmethod
    def from_constructor(
        cls,
        app_instance: FastAPI,
        api_functions: Dict[str, Tuple[EndPointProtocolFunction, List[str], BaseModel]],
    ) -> "WebFrameworkProtocol":
        """
        Constructs a FastAPIFramework instance and sets it up by adding routes.

        This method replicates the functionality of setup_app directly within
        the class as a constructor method.

        Args:
            app_instance (FastAPI): An instance of the FastAPI application.
            api_functions (Dict[str, Tuple[EndPointProtocolFunction, List[str]]]):
                A dictionary mapping route paths to endpoint functions and methods.
                Example:
                    {
                        "/example": (example_endpoint_function, ["GET", "POST"])
                    }

        Returns:
            WebFrameworkProtocol: A fully configured instance of FastAPIFramework.
        """
        # Create a new instance
        framework = cls(app_instance)

        # Add routes to the framework
        for path, (endpoint, methods, response_model) in api_functions.items():
            framework.add_route(
                path=path,
                endpoint=endpoint,
                methods=methods,
                response_model=response_model,
            )

        return framework
