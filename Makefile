.PHONY: install lint fmt test clean

install:
	pip install -e ".[dev]"

lint:
	ruff check paku tests

fmt:
	ruff format paku tests

test:
	pytest --cov=paku --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
