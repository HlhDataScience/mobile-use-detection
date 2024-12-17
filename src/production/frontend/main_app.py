"""The main API and app program entry point"""

import streamlit as st
from app import StreamlitApp
from appfuncs import inference_point, provide_class_info

user_interface = StreamlitApp()
user_interface.create_ui(
    title="Mobile Usage Classifier", message="Enter your usage details below:"
)
# Inputs
app_usage_time = st.number_input("App Usage Time (min/day)", min_value=0, step=1)
screen_on_time = st.number_input("Screen On Time (hours/day)", min_value=0.0, step=0.1)
battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0, step=1)
num_apps = st.number_input("Number of Apps Installed", min_value=0, step=1)
data_usage = st.number_input("Data Usage (MB/day)", min_value=0, step=1)

if st.button("Classify"):
    input_features = {
        "AppUsageTime_min_day": app_usage_time,
        "ScreenOnTime_hours_day": screen_on_time,
        "BatteryDrain_mAh_day": battery_drain,
        "NumberOfAppsInstalled": num_apps,
        "DataUsage_MB_day": data_usage,
    }
    class_prediction = inference_point(
        api_end_point="http://0.0.0.0:8001/predict/", input_data=input_features
    )
    additional_info = provide_class_info(class_prediction)
    st.subheader("Prediction")
    st.write(class_prediction)
    st.subheader("Additional Information")
    st.write(additional_info)
print("Streamlit initialized successfully.")
