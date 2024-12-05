"""Protocol class to ensure compatibility with different web frameworks."""

from collections.abc import Coroutine
from typing import Any, Dict, List, Protocol, Sequence, Tuple


class EndPointProtocolFunction(Protocol):
    """Protocol for API endpoint functions."""

    async def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Coroutine[Any, Any, Dict[str, Any] | Any]: ...


class WebFrameworkProtocol(Protocol):
    """Protocol class to ensure compatibility with different web frameworks."""

    def __init__(self, app):
        self.app = app

    def add_route(
        self, path: str, endpoint: EndPointProtocolFunction, methods: Sequence[str]
    ) -> None:
        """Method to add a route to the web framework."""
        pass

    def run(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        """Method to start the server."""
        pass

    @classmethod
    def from_constructor(
        cls,
        app_instance: Any,
        api_functions: Dict[str, Tuple[EndPointProtocolFunction, List[str]]],
    ) -> "WebFrameworkProtocol":
        """
        Constructs a WebFrameworkProtocol instance and sets it up by adding routes.

        Args:
            app_instance (Any): An instance of the web application framework.
            api_functions (Dict): A dictionary mapping paths to endpoints and methods.

        Returns:
            WebFrameworkProtocol: Configured framework instance.
        """
        # Create a new instance
        framework = cls(app_instance)

        # Reproduce the functionality of setup_app
        for path, (endpoint, methods) in api_functions.items():
            framework.add_route(path=path, endpoint=endpoint, methods=methods)

        return framework
