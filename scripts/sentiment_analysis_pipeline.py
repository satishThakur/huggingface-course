"""
Script to demonstrate simple sentiment analysis using HuggingFace pipelines.
"""
from src.sentiment_analysis_pipeline import analyze_sentiment_pipeline

def main():
    text = "I've been waiting for a Hugging Face course my whole life."
    result = analyze_sentiment_pipeline(text)
    
    print(f"\nText: {text}")
    print(f"Sentiment: {result[0]['label']}")
    print(f"Confidence: {result[0]['score']:.4f}")

if __name__ == "__main__":
    main()