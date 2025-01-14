# Makefile for initializing a new project

# Install Poetry dependencies
install-poetry:  ## Install Poetry dependencies
	@echo "Installing Poetry dependencies..."
	poetry install

# Activate the Poetry virtual environment
activate:  ## Activate the Poetry virtual environment
	@echo "Activating virtual environment..."
	poetry shell

# Initialize Git repository
git_initialization:  ## Initialize Git repository
	@echo "Initializing Git..."
	git init

# Install pre-commit hooks
pre-commit:  ## Install pre-commit hooks
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Install dependencies and tools, and set up the project
initialize_all: install-poetry git_initialization pre-commit activate  ## Initialize all
	@echo "Project successfully initialized."

# Help command to show available options
help:  ## Show this help message
	@echo "Available commands:"
	@echo "  make install-poetry        Install Poetry dependencies"
	@echo "  make activate              Activate virtual environment"
	@echo "  make git_initialization    Initialize Git repository"
	@echo "  make pre-commit            Install pre-commit hooks"
	@echo "  make initialize_all        Initialize everything (Poetry, Git, pre-commit)"