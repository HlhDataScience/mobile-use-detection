"""The main API and app program entry point"""

import argparse
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple

import gradio as gr
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel

from Phases.SrcProduction.api.apifuncs import (
    classify,
    get_results,
    load_classifier,
    read_root,
    results,
)
from Phases.SrcProduction.api.Frameworks import FastAPIFramework
from Phases.SrcProduction.api.validation_classes import (
    APIInfo,
    ClassifierOutput,
    ResultsDisplay,
)
from Phases.SrcProduction.app.app import (
    BlockGradioApp,
    GradioApp,
    StreamlitApp,
)
from Phases.SrcProduction.app.appfuncs import (
    gradio_inference_wrapper,
    inference_point,
    provide_class_info,
)
from Phases.SrcProduction.interfaces.WebFrameworksProtocols import (
    EndPointProtocolFunction,
)

# CONSTANTS
MODEL_PATH = "Phases/ModelsProduction/Tree_Classifier_New_v4.joblib"
# noinspection PyTypeChecker
API_CONSTRUCTOR: Dict[str, Tuple[EndPointProtocolFunction, List[str], BaseModel]] = {
    "/": (read_root, ["GET"], APIInfo),
    "/predict/": (classify, ["POST"], ClassifierOutput),
    "/results/": (results, ["GET"], ResultsDisplay),
    "/results/get_results/": (get_results, ["GET"], ResultsDisplay),
}


# Launch API
# noinspection PyShadowingNames
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager to handle the lifecycle of the FastAPI app.

    This function loads the classifier model on startup and ensures it is
    unloaded when the app shuts down.

    Args:
        app (FastAPI): The FastAPI app instance.

    Yields:
        None
    """
    try:
        classifier = load_classifier(model_path=MODEL_PATH)
        app.state.classifier = classifier
        yield
    except Exception as e:
        print(f"Error during classifier loading: {e}")
        yield
    finally:
        if hasattr(app.state, "classifier"):
            del app.state.classifier


framework = FastAPIFramework.from_constructor(
    app_instance=FastAPI(lifespan=lifespan), api_functions=API_CONSTRUCTOR
)
app = framework.app


def main():
    """
    Main entry point for the application.

    This function parses command-line arguments, selects the appropriate
    user interface (Gradio, Streamlit, or BlockGradio), and runs the
    selected UI along with the API server.

    Returns:
        None
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Run the API and App services")
    parser.add_argument(
        "--ui_type",
        type=str,
        choices=["gradio", "streamlit", "blocks"],
        default="gradio",
        help="Type of UI to launch (gradio, streamlit, or blocks).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mobile Usage Classifier",
        help="Title for the application (used in Streamlit).",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Enter your usage details below:",
        help="Message for the application (used in Streamlit).",
    )
    args = parser.parse_args()

    # UI Selection
    if args.ui_type == "gradio":
        print("Initializing Gradio...")
        app_usage_time_input = gr.Number(label="App Usage Time (min/day)")
        screen_on_time_input = gr.Number(label="Screen On Time (hours/day)")
        battery_drain_input = gr.Number(label="Battery Drain (mAh/day)")
        num_apps_input = gr.Number(label="Number of Apps Installed")
        data_usage_input = gr.Number(label="Data Usage (MB/day)")
        class_output = gr.Textbox(label="*Class of usage of mobile*")
        info_output = gr.Textbox(label="Additional Information")

        user_interface = GradioApp(
            interface=gradio_inference_wrapper,
            inputs=[
                app_usage_time_input,
                screen_on_time_input,
                battery_drain_input,
                num_apps_input,
                data_usage_input,
            ],
            outputs=[class_output, info_output],
        )
        print("Gradio initialized successfully.")
        user_interface.run()
        print("app closed successfully")
    elif args.ui_type == "streamlit":
        print("Initializing Streamlit...")
        user_interface = StreamlitApp()
        user_interface.create_ui(title=args.title, message=args.message)
        # Inputs
        app_usage_time = st.number_input(
            "App Usage Time (min/day)", min_value=0, step=1
        )
        screen_on_time = st.number_input(
            "Screen On Time (hours/day)", min_value=0.0, step=0.1
        )
        battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0, step=1)
        num_apps = st.number_input("Number of Apps Installed", min_value=0, step=1)
        data_usage = st.number_input("Data Usage (MB/day)", min_value=0, step=1)
        # Prediction
        if st.button("Classify"):
            """
            Classifies the input data when the 'Classify' button is pressed.

            This function uses the user inputs from Streamlit, makes a prediction
            using the classifier, and displays the prediction along with additional
            information.

            Args:
                None

            Returns:
                None
            """
            input_features = {
                "AppUsageTime_min_day": app_usage_time,
                "ScreenOnTime_hours_day": screen_on_time,
                "BatteryDrain_mAh_day": battery_drain,
                "NumberOfAppsInstalled": num_apps,
                "DataUsage_MB_day": data_usage,
            }
            class_prediction = inference_point(input_features)
            additional_info = provide_class_info(class_prediction)
            st.subheader("Prediction")
            st.write(class_prediction)
            st.subheader("Additional Information")
            st.write(additional_info)
        print("Streamlit initialized successfully.")
    elif args.ui_type == "blocks":
        print("Initializing Blocks UI...")
        user_interface = BlockGradioApp()
        user_interface.create_ui(
            components=[
                {"type": "markdown", "content": "## Smartphone Usage Inference"},
                {
                    "type": "number",
                    "name": "app_usage_time",
                    "label": "App Usage Time (min/day)",
                    "value": 0,
                },
                {
                    "type": "number",
                    "name": "screen_on_time",
                    "label": "Screen On Time (hours/day)",
                    "value": 0.0,
                },
                {
                    "type": "number",
                    "name": "battery_drain",
                    "label": "Battery Drain (mAh/day)",
                    "value": 0,
                },
                {
                    "type": "number",
                    "name": "num_apps",
                    "label": "Number of Apps Installed",
                    "value": 0,
                },
                {
                    "type": "number",
                    "name": "data_usage",
                    "label": "Data Usage (MB/day)",
                    "value": 0.0,
                },
            ]
        )
        print("Blocks UI initialized successfully.")
        user_interface.run()
        print("app closed successfully")


if __name__ == "__main__":
    """
    Entry point for running the FastAPI server and the selected user interface.

    This function starts the FastAPI server in a separate thread and
    runs the main application logic, including selecting and launching the UI.

    Args:
        None

    Returns:
        None
    """
    print("Starting FastAPI server...")
    threading.Thread(target=lambda: framework.run(port=8001), daemon=True).start()
    main()
