"""Module for the WebApp protocol interface"""

from typing import Any, Callable, Dict, Protocol, Sequence


class AppConstructorProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """The constructor should be callable and set up the app."""
        pass


class WebAppProtocol:
    pass
