"""
Module for interacting with the Hugging Face Hub API to list and search models.
"""
from huggingface_hub import HfApi

def list_top_models(limit=10, sort="downloads"):
    """
    Fetch top models from Hugging Face Hub based on specified criteria.
    
    Args:
        limit (int): Maximum number of models to return
        sort (str): Sorting criteria (e.g., "downloads", "likes")
        
    Returns:
        list: List of model objects with their metadata
    """
    api = HfApi()
    print('Fetching models from HuggingFace Hub...')
    models = list(api.list_models(limit=limit, sort=sort))
    print(f'Retrieved {len(models)} models')
    return models

if __name__ == "__main__":
    # For testing the module directly
    models = list_top_models(limit=5)
    for model in models:
        print("\n" + "="*50)
        print(f"Model ID: {model.id}")
        print(f"Name: {model.modelId}")
        print(f"Downloads: {model.downloads}")
        print(f"Tags: {', '.join(model.tags)}")