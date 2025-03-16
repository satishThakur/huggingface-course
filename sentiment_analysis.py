import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def analyze_sentiment(text, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get sentiment scores
    negative_score = predictions[0, 0].item()
    positive_score = predictions[0, 1].item()
    
    # Determine sentiment label
    sentiment = "positive" if positive_score > negative_score else "negative"
    confidence = positive_score if sentiment == "positive" else negative_score
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": {
            "negative": negative_score,
            "positive": positive_score
        }
    }

if __name__ == "__main__":
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