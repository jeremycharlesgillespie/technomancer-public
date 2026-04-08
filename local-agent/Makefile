.PHONY: test lint format start stop status update install clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install package in editable mode with all deps"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make lint       - Check code style (ruff + black)"
	@echo "  make format     - Auto-format code (ruff + black)"
	@echo "  make start      - Start the Discord bot"
	@echo "  make stop       - Stop the Discord bot"
	@echo "  make status     - Show bot status"
	@echo "  make update     - Start safe_update workflow (use: make update name=fix-bug)"
	@echo "  make clean      - Remove build artifacts"

# Installation
install:
	pip install -e ".[all,dev]"
	pre-commit install

# Testing
test:
	pytest -v

test-cov:
	pytest --cov=agent --cov-report=term-missing --cov-report=html

test-fast:
	pytest -x -q

# Code quality
lint:
	ruff check agent/ tests/
	black --check agent/ tests/

format:
	black agent/ tests/
	ruff check --fix agent/ tests/

# Bot management
start:
	python bot_service.py start

stop:
	python bot_service.py stop

status:
	python bot_service.py status

restart:
	python bot_service.py start

# Safe update workflow
update:
ifndef name
	$(error name is required. Usage: make update name=fix-bug)
endif
	python safe_update.py $(name)

continue:
	python safe_update.py continue

abort:
	python safe_update.py abort

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
