# Goal Classifier Model

## Overview
This is a fine-tuned DistilBERT model that classifies goals into three categories:
- **Vague**: Goals that lack specificity and measurable outcomes
- **Partial SMART**: Goals that have some SMART elements but are missing others
- **SMART**: Goals that are Specific, Measurable, Achievable, Relevant, and Time-bound

## Model Details
- Base model: `distilbert-base-uncased`
- Fine-tuned on a dataset of manually labeled goals
- Trained to recognize the characteristics of well-formed SMART goals

## Usage
You can use this model to classify goals in several ways:

### Command Line Interface
```bash
# Classify a single goal
python src/goal_classifier.py --classify "Your goal text here"

# Interactive mode
python src/goal_classifier.py --interactive

# Use existing model without retraining
python src/goal_classifier.py --skip-train --interactive
```

### Programmatic Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model and tokenizer
model_path = "./goal_classifier_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Function to predict goal type
def predict_goal_type(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    
    # Map the predicted class ID to the label
    label_mapping = {0: "vague", 1: "partial smart", 2: "smart"}
    predicted_label = label_mapping[predicted_class_id]
    
    return predicted_label

# Example usage
goal = "Complete a marathon in under 4 hours by following a 16-week training plan"
prediction = predict_goal_type(goal, model, tokenizer)
print(f"Goal classification: {prediction}")
```

## Model Training
The model was trained on a dataset of goals categorized as vague, partially SMART, or fully SMART. The training process involved:
1. Loading labeled goal data from text files
2. Tokenizing the text using the DistilBERT tokenizer
3. Fine-tuning the DistilBERT model for sequence classification
4. Evaluating the model on a validation set

## Example Classifications

### Vague Goals
- "Become a better person"
- "Improve my coding skills"

### Partially SMART Goals
- "Learn to play the guitar by practicing 30 minutes daily"
- "Read 2 books per month"

### SMART Goals
- "Complete a marathon in under 4 hours by following a 16-week training plan and running 4 times per week"
- "Lose 15 pounds by June by following a calorie-deficit diet and exercising 5 times a week, tracking progress weekly"

## Requirements
- transformers
- torch
- pandas
- numpy
- scikit-learn
- datasets
