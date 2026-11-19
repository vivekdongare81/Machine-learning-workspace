# Common tasks. Run `make help` to list them.
PYTHON ?= python
CONFIG ?= configs/train.yaml
IMAGE  ?= churn-api
TAG    ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo local)

.DEFAULT_GOAL := help
.PHONY: help install hooks lock lint format typecheck test check train score-batch serve \
        docker-build docker-run clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

install: ## Install pinned dev dependencies and this package (editable)
	$(PYTHON) -m pip install -r requirements-dev.lock
	$(PYTHON) -m pip install --no-deps -e .

hooks: ## Install git pre-commit hooks (run once after `git init`)
	pre-commit install

lock: ## Re-pin dependencies from pyproject.toml into the lock files (needs uv)
	uv pip compile pyproject.toml --universal --python-version 3.10 \
		--custom-compile-command "make lock" -o requirements.lock
	uv pip compile pyproject.toml --universal --python-version 3.10 --extra dev \
		--custom-compile-command "make lock" -o requirements-dev.lock

lint: ## Check style and common bugs (ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Auto-format and auto-fix lint issues
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck: ## Static type check (mypy)
	$(PYTHON) -m mypy

test: ## Run tests with coverage (fails under 80%)
	$(PYTHON) -m pytest --cov --cov-report=term-missing

check: lint typecheck test ## Everything CI runs

train: ## Train, evaluate, gate and publish a model (CONFIG=...)
	$(PYTHON) -m churn_ml.train --config $(CONFIG)

score-batch: ## Batch-score a CSV: make score-batch INPUT=in.csv OUTPUT=out.csv
	$(PYTHON) -m churn_ml.predict --input $(INPUT) --output $(OUTPUT)

serve: ## Run the API locally with auto-reload
	$(PYTHON) -m uvicorn churn_ml.api:app --host 127.0.0.1 --port 8000 --reload

docker-build: ## Build the serving image (train first: it bakes in artifacts/)
	docker build -t $(IMAGE):$(TAG) -t $(IMAGE):latest .

docker-run: ## Run the serving image on localhost:8000
	docker run --rm -p 127.0.0.1:8000:8000 $(IMAGE):latest

clean: ## Remove caches and build leftovers
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
