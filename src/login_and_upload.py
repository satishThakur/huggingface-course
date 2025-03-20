"""
Script to login to Hugging Face and upload the model in one step.
"""
import os
import argparse
from huggingface_hub import HfApi, login, create_repo

def login_and_upload(model_path, repo_name, token, private=False):
    """
    Login to Hugging Face and upload a model to the Model Hub.
    
    Args:
        model_path (str): Path to the model directory
        repo_name (str): Name of the repository to create/use on the Hub
        token (str): Hugging Face API token
        private (bool): Whether the repository should be private
    """
    # Login to Hugging Face
    print("Logging in to Hugging Face...")
    login(token=token, add_to_git_credential=True)
    print("Login successful!")
    
    # Create API instance
    api = HfApi(token=token)
    
    # Create a new repository if it doesn't exist
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
    parser = argparse.ArgumentParser(description="Login to Hugging Face and upload a model")
    parser.add_argument("--model_path", type=str, default="./goal_classifier_model", 
                        help="Path to the model directory")
    parser.add_argument("--repo_name", type=str, required=True, 
                        help="Name of the repository on the Hub (username/repo-name)")
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", 
                        help="Whether the repository should be private")
    
    args = parser.parse_args()
    
    login_and_upload(args.model_path, args.repo_name, args.token, args.private)
