"""
Script to list popular models from Hugging Face Hub.
"""
# Import directly from the module since it's been moved to src
from src.model_listing import list_top_models

def main():
    # List top 10 models from HuggingFace Hub
    models = list_top_models(limit=10)
    
    # Display model information
    for model in models:
        print("\n" + "="*50)
        print(f"Model ID: {model.id}")
        print(f"Name: {model.modelId}")
        print(f"Last Modified: {model.lastModified}")
        print(f"Tags: {', '.join(model.tags)}")
        print(f"Pipeline Tag: {model.pipeline_tag if model.pipeline_tag else 'None'}")
        print(f"Private: {model.private}")
        print(f"Downloads: {model.downloads}")
        print(f"Likes: {model.likes}")

if __name__ == "__main__":
    main()