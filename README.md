# Hugging Face Course Project

This project demonstrates various functionalities using Hugging Face's transformers library and model hub.

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
poetry run python3 model_listing.py
```
Lists top 10 models from Hugging Face Hub sorted by downloads.

### 2. Run Sentiment Analysis (Pipeline Version)
```bash
poetry run python3 sentiment_analysis_pipeline.py
```
Quick analysis using Hugging Face pipeline approach.

### 3. Run Sentiment Analysis (Detailed Version)
```bash
poetry run python3 sentiment_analysis.py
```
Detailed implementation with custom confidence scores and multiple test examples.

## Project Components

### 1. Model Listing (`model_listing.py`)
Lists top models from Hugging Face Hub sorted by downloads. Shows:
- Model ID and name
- Tags and pipeline type
- Download and like counts
- Last modified date

### 2. Sentiment Analysis

Two implementations available:

#### a. Using Pipeline (`sentiment_analysis_pipeline.py`)
Quick and simple approach using Hugging Face pipelines.

#### b. Custom Implementation (`sentiment_analysis.py`)
Detailed implementation showing model loading and inference.

## Dependencies
- transformers
- torch
- huggingface-hub
- datasets
- tokenizers

All dependencies are managed through Poetry and specified in `pyproject.toml`. 