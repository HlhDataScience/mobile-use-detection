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

from Phases.SrcProduction.app.appfuncs import (
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
    Implementation of `AppInterfaceProtocol` for Gradio-style blocks.

    This class provides methods for creating and rendering a block-based UI.

    Attributes:
        block (gr.Blocks): The Gradio Blocks object for managing the UI.

    Methods:
        - create_ui(*args, **kwargs): Configures the block-based user interface.
        - run(): Renders the block and launches the application.
    """

    def __init__(self):
        """
        Initializes the BlockGradioApp.

        Sets up a Gradio Blocks object for managing block-based components.
        """
        self.block = gr.Blocks()

    def create_ui(self, *args: Any, **kwargs: Any) -> None:
        """
        Configures the block-based user interface.

        Args:
            *args: Positional arguments for UI configuration.
            **kwargs: Keyword arguments for component customization.

        Keyword Arguments:
            - components (list[dict]): A list of dictionaries defining UI components.
              Example:
                [{"type": "textbox", "label": "Enter your name"},
                 {"type": "button", "label": "Submit"}]

        Returns:
            None
        """
        with self.block:
            gr.Markdown("## Smartphone Usage Inference")
            AppUsageTime = gr.Number(label="App Usage Time (min/day)", value=0)
            ScreenOnTime = gr.Number(label="Screen On Time (hours/day)", value=0.0)
            BatteryDrain = gr.Number(label="Battery Drain (mAh/day)", value=0)
            NumApps = gr.Number(label="Number of Apps Installed", value=0)
            DataUsage = gr.Number(label="Data Usage (MB/day)", value=0.0)
            class_output = gr.Textbox(label="Inference Result", interactive=False)
            additional_info_output = gr.Textbox(
                label="Additional info to interpreted the result."
            )

            def infer_and_provide_info(
                AppUsageTime, ScreenOnTime, BatteryDrain, NumApps, DataUsage
            ):
                inference_result = gradio_inference(
                    AppUsageTime, ScreenOnTime, BatteryDrain, NumApps, DataUsage
                )
                additional_info = provide_class_info(inference_result)
                return inference_result, additional_info

            gr.Button("Submit").click(
                infer_and_provide_info,
                inputs=[AppUsageTime, ScreenOnTime, BatteryDrain, NumApps, DataUsage],
                outputs=[class_output, additional_info_output],
            )

    def run(self) -> None:
        """
        Renders the block and launches the application.

        Returns:
            None
        """
        self.block.launch()
