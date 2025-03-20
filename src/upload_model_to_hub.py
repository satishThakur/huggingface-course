"""
Script to upload the fine-tuned goal classifier model to the Hugging Face Model Hub.
"""
import os
import argparse
from huggingface_hub import HfApi, create_repo

def upload_model_to_hub(model_path, repo_name, private=False, token=None):
    """
    Upload a model to the Hugging Face Model Hub.
    
    Args:
        model_path (str): Path to the model directory
        repo_name (str): Name of the repository to create/use on the Hub
        private (bool): Whether the repository should be private
        token (str): Hugging Face API token
    """
    # Create a new repository if it doesn't exist
    api = HfApi(token=token)
    
    try:
        print(f"Creating repository: {repo_name}")
        create_repo(repo_name, private=private, token=token, exist_ok=True)
        print(f"Repository created: {repo_name}")
    except Exception as e:
        print(f"Repository creation error (may already exist): {e}")
    
    # Upload the model files
    print(f"Uploading model files from {model_path} to {repo_name}")
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        commit_message="Upload goal classifier model",
    )
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model to the Hugging Face Model Hub")
    parser.add_argument("--model_path", type=str, default="./goal_classifier_model", 
                        help="Path to the model directory")
    parser.add_argument("--repo_name", type=str, required=True, 
                        help="Name of the repository on the Hub (username/repo-name)")
    parser.add_argument("--private", action="store_true", 
                        help="Whether the repository should be private")
    parser.add_argument("--token", type=str, 
                        help="Hugging Face API token (if not provided, will use the HUGGING_FACE_HUB_TOKEN environment variable)")
    
    args = parser.parse_args()
    
    # Use the provided token or get it from environment variable
    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Warning: No token provided. You need to be logged in with `huggingface-cli login` or provide a token.")
    
    upload_model_to_hub(args.model_path, args.repo_name, args.private, token)
