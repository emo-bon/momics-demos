.PHONY: help install dev-install test coverage lint format clean docs build

# TOOL ?= uv
TOOL ?= $(shell command -v uv >/dev/null 2>&1 && echo uv || echo poetry)

# -------------------------
# Tool abstraction
# -------------------------

ifeq ($(TOOL),uv)
	INSTALL = uv sync
	DEV_INSTALL = uv sync --extra testing && \
		uv run python -m ipykernel install --user --name momics-demos --display-name "momics-demos"
	RUN = uv run
	BUILD = uv build
endif

ifeq ($(TOOL),poetry)
	INSTALL = poetry install
	DEV_INSTALL = poetry install --with testing && \
		poetry run pip install git+https://github.com/fair-ease/py-udal-mgo.git@main && \
		poetry run python -m ipykernel install --user --name momics-demos --display-name "momics-demos"
	RUN = poetry run
	BUILD = poetry build
endif

# -------------------------
# Commands
# -------------------------

help:  ## Show this help message
	@echo 'Usage: make [target] TOOL=[uv|poetry]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install the package
	$(INSTALL)

dev-install:  ## Install with development dependencies
	$(DEV_INSTALL)

test:  ## Run tests
	$(RUN) pytest test/

coverage:  ## Run tests with coverage
	$(RUN) pytest --cov=. --cov-report=xml --cov-report=term test/

lint:  ## Run linters
	$(RUN) ruff check src test
	$(RUN) mypy src

format:  ## Format code
	$(RUN) ruff format src test
	$(RUN) ruff check --fix src test

# docs:  ## Build documentation
# 	cd docs && $(RUN) sphinx-build -b html source _build/html

build:  ## Build the package
	$(BUILD)

clean:  ## Clean artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

.DEFAULT_GOAL := help