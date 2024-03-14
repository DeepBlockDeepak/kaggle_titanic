## Guideline

This project uses `isort`, `black`, `ruff`, and Python's built-in `unittest` framework to maintain code quality and use automated tests. These tools have been integrated into the project's development environment via Poetry (see [pyproject.toml](./pyproject.toml)), alongside custom scripts for their execution.

To install these tools along with the project's dependencies, ensure you have Poetry installed, then run:

```bash
poetry install
```


### Running All Checks & Testing
To format, lint, and run tests sequentially with a single command, you can use:

```bash
poetry run all-checks
```

Go-to for testing everything in one go, before pushing changes. Should indicate passing of checks in upstream CI workflow.


### Running Code Formatters and Linters

Before committing your code, run the following command to automatically format your code and check for linting errors:

```bash
poetry run format
poetry run lint
```
These commands utilize isort and black for formatting, and ruff for linting, respectively, as configured in the project's pyproject.toml.

### Running Automated Tests
The tests/ directory contains automated tests for the project. To run these tests, execute the following command from the project root:

```bash
poetry run test
```
This command uses unittest framework to discover and run tests, i.e., `python -m unittest discover -s tests` in the `tests/` directory.


