# HuggingFace Course Codebase Guidelines

## Project Structure
```
huggingface-course/
├── data/           # Datasets and data files
├── models/         # Saved model checkpoints
├── notebooks/      # Jupyter notebooks for exploration
├── scripts/        # Runnable example scripts
└── src/            # Reusable Python modules
```

## Commands
- Run module: `poetry run python -m scripts.module_name`
- Run script directly: `poetry run python scripts/script_name.py`
- Add dependency: `poetry add package_name`
- Install dependencies: `poetry install`
- Setup/activate env: `poetry shell`
- Start Jupyter: `poetry run jupyter notebook`

## Notebook Development
- Use template.ipynb as a starting point for new notebooks
- Add sys.path.append("..") to import from src package
- Save trained models to models/ directory
- Save datasets to data/ directory

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

## Development Process
1. Explore ideas in notebooks
2. Extract reusable code to src/ modules
3. Create runnable examples in scripts/
4. Document functionality in README.md