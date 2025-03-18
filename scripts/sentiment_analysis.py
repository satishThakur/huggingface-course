"""
Script to demonstrate detailed sentiment analysis using HuggingFace models.
"""
from src.sentiment_analysis import analyze_sentiment

def main():
    # Test with some example texts
    texts = [
        "I love this product, it's amazing!",
        "This was a terrible experience, I'm very disappointed.",
        "The movie was okay, nothing special."
    ]
    
    print("Sentiment Analysis Results:")
    print("==========================")
    
    for text in texts:
        result = analyze_sentiment(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.4f})")
        print(f"Scores: Positive: {result['scores']['positive']:.4f}, Negative: {result['scores']['negative']:.4f}")

if __name__ == "__main__":
    main()