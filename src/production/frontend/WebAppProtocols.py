"""
WebAppProtocols Module

This module contains the `AppInterfaceProtocol`, which serves as a
blueprint for creating applications using various frameworks such
as Gradio and Streamlit. The protocol ensures that all implementations
adhere to a common interface for creating and running apps.

Classes:
    - AppInterfaceProtocol: A protocol defining methods for creating
      a user interface and running the application.

Usage:
    Implement this protocol in classes to ensure compatibility with
    the specified interface.
"""

from typing import Any, Optional, Protocol


class AppInterfaceProtocol(Protocol):
    """
    Interface for easy swapping of app libraries.

    This protocol defines a structure for applications using third-party
    frameworks such as Gradio or Streamlit. It ensures a consistent interface
    for creating and running apps.

    Methods:
        - create_ui(*args, **kwargs): Sets up the user interface. Can take
          additional arguments or keywords specific to the implementation.
        - run(): Starts the application.
    """

    def create_ui(self, *args: Any, **kwargs: Any) -> Optional[None]:
        """
        Sets up the user interface for the application.

        Args:
            *args: Positional arguments specific to the implementation.
            **kwargs: Keyword arguments specific to the implementation.

        Returns:
            None
        """
        ...

    def run(self) -> None:
        """
        Starts the application.

        This method is responsible for launching the app, whether through
        a local server or another mechanism.

        Returns:
            None
        """
        ...
