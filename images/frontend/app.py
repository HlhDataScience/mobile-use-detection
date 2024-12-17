"""
app Module

This module implements the `AppInterfaceProtocol` for various frameworks,
including Gradio, Streamlit, and block-based Gradio applications.

Classes:
    - GradioApp: Implements `AppInterfaceProtocol` for Gradio.
    - StreamlitApp: Implements `AppInterfaceProtocol` for Streamlit.
    - BlockGradioApp: Implements `AppInterfaceProtocol` for Gradio blocks.

Usage:
    Select the desired implementation, configure the UI using `create_ui`,
    and start the app using `run`.
"""

from typing import Any

import streamlit as st


class StreamlitApp:
    """
    Implementation of `AppInterfaceProtocol` for Streamlit.

    This class provides methods for creating and running a Streamlit app.

    Attributes:
        st (module): The Streamlit module.

    Methods:
        - create_ui(*args, **kwargs): Configures the Streamlit UI.
        - run(): Launches the Streamlit application.
    """

    def __init__(self):
        """
        Initializes the StreamlitApp.

        Imports the Streamlit module for internal use.
        """

        self.st = st

    def create_ui(self, *args: Any, **kwargs: Any) -> None:
        """
        Configures the Streamlit user interface.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments for UI customization.

        Keyword Arguments:
            - title (str): App title (default: "Streamlit App").
            - message (str): Welcome message (default: "Welcome!").

        Returns:
            None
        """
        title = kwargs.get("title", "Streamlit App")
        message = kwargs.get("message", "Welcome!")
        self.st.title(title)
        self.st.write(message)

    def run(self) -> None:
        """
        Streamlit apps are run by executing `streamlit run script_name.py`.

        Returns:
            None
        """
        print("Run `streamlit run app.py` to launch the application.")
