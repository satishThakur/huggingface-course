# HuggingFace Course Codebase Guidelines

## Commands
- Run script: `poetry run python3 script_name.py`
- Add dependency: `poetry add package_name`
- Install dependencies: `poetry install`
- Setup/activate env: `poetry shell` 

## Code Style
- Follow PEP 8 standards for Python code
- Use docstrings for functions and classes
- Import order: standard library, third-party, local modules
- Type hints encouraged for function parameters and returns
- Variable naming: lowercase with underscores (snake_case)
- Class naming: CamelCase
- Error handling: use try/except blocks appropriately
- Prefer explicit over implicit code
- Keep functions focused on single responsibility
- Use descriptive variable names