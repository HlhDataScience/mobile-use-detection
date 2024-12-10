# Define variables
PYTHON := python
API := fastapi
STREAMLIT := streamlit
MAIN_SCRIPT := main.py
APP_SCRIPT := main_app.py
LOG_DIR := logs

.PHONY: all transformation training clean api_dev api_run gradio_app_run streamlit_app_run blocks_app_run help

# Default target
all: run_all

# Run the entire program (both pipelines)
run_all:
	$(PYTHON) $(MAIN_SCRIPT) --pipeline all

# Run only the transformation pipeline
transformation:
	$(PYTHON) $(MAIN_SCRIPT) --pipeline transformation

# Run only the training pipeline
training:
	$(PYTHON) $(MAIN_SCRIPT) --pipeline training

# Clean up logs
clean:
	@echo "Cleaning up logs..."
	rm -rf $(LOG_DIR)/*
	@echo "Logs cleaned."

api_dev:
	$(API) dev $(APP_SCRIPT) --port 8001 2>&1 | tee -a logs/api_dev.log

api_run:
	$(API) run $(APP_SCRIPT) --port 8001 2>&1 | tee -a logs/api_run.log

gradio_app_run:
	@echo "Starting Gradio app..."
	$(PYTHON) main_app.py --ui_type gradio

streamlit_app_run:
	$(STREAMLIT) run main_app.py -- --ui_type streamlit

blocks_app_run:
	$(PYTHON) main_app.py --ui_type blocks

# Display help
help:
	@echo "Makefile for managing the program"
	@echo
	@echo "Available targets:"
	@echo "  run_all          Run both the transformation and training pipelines."
	@echo "  transformation   Run only the transformation pipeline."
	@echo "  training         Run only the training pipeline."
	@echo "  clean            Remove all logs from the 'logs' directory."
	@echo "  api_dev          Starts FASTAPI in development mode."
	@echo "  api_run          Starts FASTAPI in production mode."
	@echo "  gradio_app_run   Start the Gradio app."
	@echo "  streamlit_app_run Start the Streamlit app."
	@echo "  blocks_app_run   Start the Blocks app."
	@echo "  help             Display this help message."
