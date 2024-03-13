import subprocess


def run():
    """Lints the project's Python code using `ruff`.

    Ensures code adheres to defined standards and catches potential issues.
    """
    subprocess.run(["ruff", "."], check=True)
