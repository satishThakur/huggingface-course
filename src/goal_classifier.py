"""
Module for training a text classification model to classify goals as 'vague', 'partial smart', or 'smart'.
This uses the Hugging Face transformers library to fine-tune a pre-trained language model.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import argparse

# Define our label mapping
LABEL_MAPPING = {
    "vague": 0,
    "partial smart": 1,
    "smart": 2
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def load_goals_from_file(file_path):
    """
    Load goals from a text file.
    
    Args:
        file_path (str): Path to the text file containing goals
        
    Returns:
        list: List of goals as strings
    """
    with open(file_path, 'r') as f:
        goals = [line.strip() for line in f.readlines() if line.strip()]
    return goals

def create_sample_data():
    """
    Load goal classification data from text files.
    
    Returns:
        pandas.DataFrame: DataFrame with 'text' and 'label' columns
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vague_goals_file = os.path.join(base_dir, 'goal_data', 'vague_goals.txt')
    partial_smart_goals_file = os.path.join(base_dir, 'goal_data', 'partial_smart_goals.txt')
    smart_goals_file = os.path.join(base_dir, 'goal_data', 'smart_goals.txt')
    
    # Load goals from files
    vague_goals = load_goals_from_file(vague_goals_file)
    partial_smart_goals = load_goals_from_file(partial_smart_goals_file)
    smart_goals = load_goals_from_file(smart_goals_file)
    
    print(f"Loaded {len(vague_goals)} vague goals")
    print(f"Loaded {len(partial_smart_goals)} partially SMART goals")
    print(f"Loaded {len(smart_goals)} SMART goals")
    
    # Create a list of tuples with (text, label)
    data = []
    for goal in vague_goals:
        data.append((goal, "vague"))
    for goal in partial_smart_goals:
        data.append((goal, "partial smart"))
    for goal in smart_goals:
        data.append((goal, "smart"))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["text", "label"])
    return df

def prepare_datasets(df):
    """
    Prepare training and validation datasets.
    
    Args:
        df (pandas.DataFrame): DataFrame with 'text' and 'label' columns
        
    Returns:
        tuple: (train_dataset, val_dataset) as Hugging Face datasets
    """
    # Convert labels to numeric values
    df["label_id"] = df["label"].map(LABEL_MAPPING)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    return train_dataset, val_dataset

def tokenize_data(dataset, tokenizer):
    """
    Tokenize the text data.
    
    Args:
        dataset (datasets.Dataset): Dataset to tokenize
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
        
    Returns:
        datasets.Dataset: Tokenized dataset
    """
    def tokenize_function(examples):
        # Tokenize the texts and return as a dictionary
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        # Add the labels
        tokenized["labels"] = examples["label_id"]
        return tokenized
    
    # Remove columns that are not needed for training
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text", "label", "__index_level_0__"]
    )
    
    # Set the format to PyTorch tensors
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset

def train_model(train_dataset, val_dataset, model_name="distilbert-base-uncased", num_labels=3):
    """
    Train a text classification model.
    
    Args:
        train_dataset (datasets.Dataset): Training dataset
        val_dataset (datasets.Dataset): Validation dataset
        model_name (str): Name of the pre-trained model to use
        num_labels (int): Number of classification labels
        
    Returns:
        transformers.PreTrainedModel: Trained model
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Tokenize datasets
    tokenized_train = tokenize_data(train_dataset, tokenizer)
    tokenized_val = tokenize_data(val_dataset, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",  
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )
    
    # Train the model
    print("Training the model...")
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model and tokenizer
    model_save_path = "./goal_classifier_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return model, tokenizer

def predict_goal_type(text, model, tokenizer):
    """
    Predict the goal type for a given text.
    
    Args:
        text (str): The goal text to classify
        model (transformers.PreTrainedModel): Trained model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
        
    Returns:
        str: Predicted goal type ('vague', 'partial smart', or 'smart')
    """
    # Get the device that the model is on
    device = model.device
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
    
    # Map the predicted class ID to the label
    predicted_label = REVERSE_LABEL_MAPPING[predicted_class_id]
    
    return predicted_label

def evaluate_model_with_examples(model, tokenizer):
    """
    Evaluate the model with some example goals.
    
    Args:
        model (transformers.PreTrainedModel): Trained model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
    """
    example_goals = [
        "Become a better person",
        "Learn to play the guitar by practicing 30 minutes daily",
        "Complete a marathon in under 4 hours by following a 16-week training plan and running 4 times per week",
        "Improve my coding skills",
        "Read 2 books per month",
        "Lose 15 pounds by June by following a calorie-deficit diet and exercising 5 times a week, tracking progress weekly"
    ]
    
    print("\nEvaluating model with example goals:")
    for goal in example_goals:
        prediction = predict_goal_type(goal, model, tokenizer)
        print(f"Goal: '{goal}'")
        print(f"Prediction: {prediction}\n")

def interactive_classification(model, tokenizer):
    """
    Allow users to interactively classify their own goals.
    
    Args:
        model (transformers.PreTrainedModel): Trained model
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer
    """
    print("\n" + "="*50)
    print("Goal Classification Interactive Mode")
    print("="*50)
    print("Type 'exit' to quit the interactive mode.")
    
    while True:
        user_goal = input("\nEnter a goal to classify: ")
        if user_goal.lower() == 'exit':
            print("Exiting interactive mode.")
            break
        
        if not user_goal.strip():
            print("Please enter a valid goal.")
            continue
        
        prediction = predict_goal_type(user_goal, model, tokenizer)
        
        print(f"\nClassification: {prediction.upper()}")
        
        if prediction == "vague":
            print("\nThis goal is too general and lacks specific details.")
            print("To make it SMART, consider adding specifics about:")
            print("- What exactly you want to achieve (Specific)")
            print("- How you'll measure success (Measurable)")
            print("- A realistic timeframe (Time-bound)")
        elif prediction == "partial smart":
            print("\nThis goal has some SMART elements but could be improved.")
            print("Consider adding:")
            print("- A clear timeframe for completion")
            print("- How you'll track progress")
            print("- Why this goal matters to you (Relevant)")
        else:  # smart
            print("\nThis is a well-defined SMART goal with specific details!")
            print("It includes:")
            print("- A clear objective (Specific)")
            print("- Ways to measure progress (Measurable)")
            print("- A realistic approach (Achievable)")
            print("- Relevance to your broader objectives (Relevant)")
            print("- A defined timeframe (Time-bound)")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Goal Classification using DistilBERT')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--classify', type=str, help='Classify a single goal and exit')
    parser.add_argument('--skip-train', action='store_true', help='Skip training and use existing model')
    args = parser.parse_args()
    
    # Create sample data if not skipping training
    if not args.skip_train:
        print("Creating sample data...")
        data = create_sample_data()
        print(f"Created dataset with {len(data)} examples")
        
        # Prepare datasets
        print("Preparing datasets...")
        train_dataset, val_dataset = prepare_datasets(data)
    
    # Check if model already exists
    model_path = "./goal_classifier_model"
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Loading existing model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Only train if not skipping training
        if args.skip_train:
            print("Error: Model does not exist and --skip-train was specified.")
            exit(1)
        # Train model
        print("Training the model...")
        model, tokenizer = train_model(train_dataset, val_dataset)
    
    # If a single goal was provided for classification
    if args.classify:
        prediction = predict_goal_type(args.classify, model, tokenizer)
        print(f"\nGoal: '{args.classify}'")
        print(f"Classification: {prediction.upper()}")
        exit(0)
    
    # Evaluate with examples
    evaluate_model_with_examples(model, tokenizer)
    
    # Start interactive mode if requested
    if args.interactive:
        try:
            interactive_classification(model, tokenizer)
        except EOFError:
            print("\nInteractive mode not available in this environment.")
            print("Use --classify 'your goal' to classify a single goal instead.")
