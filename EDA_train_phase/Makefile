# Define variables
PYTHON := python
MAIN_SCRIPT := main.py
LOG_DIR := logs

.PHONY: all transformation training clean help

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

# Display help
help:
	@echo "Makefile for managing the program"
	@echo
	@echo "Available targets:"
	@echo "  run_all        Run both the transformation and training pipelines."
	@echo "  transformation Run only the transformation pipeline."
	@echo "  training       Run only the training pipeline."
	@echo "  clean          Remove all logs from the 'logs' directory."
	@echo "  help           Display this help message."
