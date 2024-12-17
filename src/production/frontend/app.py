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

from typing import Any, Callable

import gradio as gr
import streamlit as st

from src.production.frontend.appfuncs import (
    gradio_inference,
    provide_class_info,
)


class GradioApp:
    """
    Implementation of `AppInterfaceProtocol` for Gradio.

    This class provides methods for creating and running a Gradio-based app.

    Attributes:
        app (gr.Interface): The Gradio interface object.

    Methods:
        - create_ui(*args, **kwargs): Configures the Gradio user interface.
        - run(): Launches the Gradio application.
    """

    def __init__(self, interface: Callable, inputs: Any, outputs: Any):
        """
        Initializes the GradioApp.

        Args:
            interface (Callable): The function to process user input.
            inputs (Any): Input components for the Gradio interface.
            outputs (Any): Output components for the Gradio interface.
        """
        self.app = gr.Interface(fn=interface, inputs=inputs, outputs=outputs)

    def create_ui(self, *args: Any, **kwargs: Any) -> None:
        """
        Configures the Gradio user interface.

        Args:
            *args: Positional arguments (unused for Gradio).
            **kwargs: Keyword arguments (unused for Gradio).

        Returns:
            None
        """
        pass  # Gradio's UI is pre-configured in the constructor.

    def run(self) -> None:
        """
        Launches the Gradio application.

        Returns:
            None
        """
        self.app.launch()


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


class BlockGradioApp:
    """
    Implementation of `AppInterfaceProtocol` for Gradio blocks.

    This class provides methods for creating and running a Block-based Gradio app.

    Attributes:
        block (gr.Blocks): The Gradio Blocks object.

    Methods:
        - create_ui(*args, **kwargs): Configures the Gradio blocks user interface.
        - run(): Launches the Gradio blocks application.
    """

    def __init__(self):
        """
        Initializes the BlockGradioApp.

        Creates an instance of `gr.Blocks` for building the UI using Gradio blocks.
        """
        self.block = gr.Blocks()

    def create_ui(self, *args: Any, **kwargs: Any) -> None:
        """
        Configures the Gradio blocks user interface.

        Args:
            *args: Positional arguments (unused for BlockGradio).
            **kwargs: Keyword arguments for UI customization.

        Keyword Arguments:
            - components (list): List of components to include in the UI.
                Each component should be a dictionary with the type and necessary values.

        Returns:
            None
        """
        components = kwargs.get("components", [])

        with self.block:
            inputs = {}
            for component in components:
                if component["type"] == "number":
                    inputs[component["name"]] = gr.Number(
                        label=component["label"], value=component.get("value", 0)
                    )
                elif component["type"] == "markdown":
                    inputs[component["content"]] = gr.Markdown(
                        value=component["content"]
                    )
                # You can add more component types here if needed

            class_output = gr.Textbox(label="Class Prediction", interactive=False)
            additional_info_output = gr.Textbox(
                label="Additional Information", interactive=False
            )

            def infer_and_provide_info(*input_data):
                """
                Runs inference and provides additional information for the prediction.

                Args:
                    *input_data: Input values to be processed for the prediction.

                Returns:
                    tuple: The class prediction and additional information.
                """
                input_data = [i for i in input_data if isinstance(i, (int, float))]
                class_prediction = gradio_inference(
                    input_data[0],
                    input_data[1],
                    input_data[2],
                    input_data[3],
                    input_data[4],
                )
                additional_info = provide_class_info(class_prediction)
                return class_prediction, additional_info

            gr.Button("Submit").click(
                infer_and_provide_info,
                inputs=list(inputs.values()),
                outputs=[class_output, additional_info_output],
            )

    def run(self) -> None:
        """
        Launches the Gradio blocks application.

        Starts the Gradio Blocks interface in the current environment.

        Returns:
            None
        """
        self.block.launch()
