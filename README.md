# Hugging Face Course Project

This project demonstrates various functionalities using Hugging Face's transformers library and model hub.

## Project Structure

```
huggingface-course/
├── data/           # Datasets and data files
├── models/         # Saved model checkpoints
├── notebooks/      # Jupyter notebooks for exploration
├── scripts/        # Runnable example scripts
└── src/            # Reusable Python modules
```

## Setup

1. Make sure you have Python 3.10+ installed
2. Install Poetry (package manager)
3. Install dependencies:
```bash
poetry install
```

## Running the Scripts

### 1. List Popular Models
```bash
poetry run python3 -m scripts.list_models
```
Lists top 10 models from Hugging Face Hub sorted by downloads.

### 2. Run Sentiment Analysis (Pipeline Version)
```bash
poetry run python3 -m scripts.sentiment_analysis_pipeline
```
Quick analysis using Hugging Face pipeline approach.

### 3. Run Sentiment Analysis (Detailed Version)
```bash
poetry run python3 -m scripts.sentiment_analysis
```
Detailed implementation with custom confidence scores and multiple test examples.

## Project Components

### 1. Model Listing (src/model_listing.py)
Lists top models from Hugging Face Hub sorted by downloads. Shows:
- Model ID and name
- Tags and pipeline type
- Download and like counts
- Last modified date

### 2. Sentiment Analysis

Two implementations available:

#### a. Using Pipeline (src/sentiment_analysis_pipeline.py)
Quick and simple approach using Hugging Face pipelines.

#### b. Custom Implementation (src/sentiment_analysis.py)
Detailed implementation showing model loading and inference.

## Notebooks

Exploratory Jupyter notebooks are in the `notebooks/` directory for interactive learning.

## Dependencies
- transformers
- torch
- huggingface-hub
- datasets
- tokenizers

All dependencies are managed through Poetry and specified in `pyproject.toml`.