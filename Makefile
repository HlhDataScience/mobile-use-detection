# Define variables
PYTHON := python
API := fastapi
MAIN_SCRIPT := main.py
APP_SCRIPT := main_app.py
LOG_DIR := logs

.PHONY: all transformation training clean api_dev api_run help

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
	$(API) dev $(APP_SCRIPT) --port 8001

api_run:
	$(API) run $(APP_SCRIPT) --port 8001
# Display help
help:
	@echo "Makefile for managing the program"
	@echo
	@echo "Available targets:"
	@echo "  run_all        Run both the transformation and training pipelines."
	@echo "  transformation Run only the transformation pipeline."
	@echo "  training       Run only the training pipeline."
	@echo "  clean          Remove all logs from the 'logs' directory."
	@echo "  api_dev        starts FASTAPI dev framework."
	@echo "  api_run        Starts FastApi production mode."
	@echo "  help           Display this help message."
