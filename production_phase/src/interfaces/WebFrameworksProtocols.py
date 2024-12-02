from typing import Callable, Protocol, Sequence


class WebFrameworkProtocol(Protocol):
    """Protocol class to ensure compatibility with different web frameworks."""

    def add_route(self, path: str, endpoint: Callable, methods: Sequence[str]) -> None:
        """Method to add a route to the web framework."""
        pass

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Method to start the server."""
        pass
