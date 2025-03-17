"""
Module for simple sentiment analysis using HuggingFace pipeline API.
"""
from transformers import pipeline

def analyze_sentiment_pipeline(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Analyze sentiment of text using HuggingFace pipeline API.
    
    Args:
        text (str): Text to analyze
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        list: List containing dictionary with sentiment label and score
    """
    classifier = pipeline(task="sentiment-analysis", model=model_name)
    result = classifier(text)
    return result

if __name__ == "__main__":
    # For testing the module directly
    text = "I've been waiting for a Hugging Face course my whole life."
    result = analyze_sentiment_pipeline(text)
    print(result)

