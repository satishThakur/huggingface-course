"""
Gradio app for the Goal Classifier model.
This app provides a simple interface to classify goals as 'vague', 'partial smart', or 'smart'.
"""
import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Define label mapping
LABEL_MAPPING = {
    0: "vague",
    1: "partial smart",
    2: "smart"
}

# Feedback messages based on classification
FEEDBACK = {
    "vague": """
        This goal is too general and lacks specific details.
        To make it SMART, consider adding specifics about:
        - What exactly you want to achieve (Specific)
        - How you'll measure success (Measurable)
        - A realistic timeframe (Time-bound)
    """,
    "partial smart": """
        This goal has some SMART elements but could be improved.
        Consider adding:
        - A clear timeframe for completion
        - How you'll track progress
        - Why this goal matters to you (Relevant)
    """,
    "smart": """
        This is a well-defined SMART goal with specific details!
        It includes:
        - A clear objective (Specific)
        - Ways to measure progress (Measurable)
        - A realistic approach (Achievable)
        - Relevance to your broader objectives (Relevant)
        - A defined timeframe (Time-bound)
    """
}

# Replace with your actual model name on Hugging Face
HF_MODEL_NAME = "satish001/goal-classifier"

# Load the model and tokenizer from Hugging Face Hub
@gr.load(cache_examples=True)
def load_model():
    """Load the goal classifier model and tokenizer."""
    print(f"Loading model from Hugging Face Hub: {HF_MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    return model, tokenizer

# Load the model and tokenizer
print("Loading model...")
model, tokenizer = load_model()
print("Model loaded successfully!")

def predict_goal_type(text):
    """Predict the goal type for a given text."""
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
    predicted_label = LABEL_MAPPING[predicted_class_id]
    
    return predicted_label

def classify_goal(goal_text):
    """Classify a goal and provide feedback."""
    if not goal_text.strip():
        return "Please enter a goal to classify.", ""
    
    # Classify the goal
    prediction = predict_goal_type(goal_text)
    
    # Get feedback based on classification
    feedback = FEEDBACK[prediction]
    
    # Format the result
    result = f"Classification: {prediction.upper()}"
    
    return result, feedback

# Create the Gradio interface
with gr.Blocks(title="Goal Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SMART Goal Classifier")
    gr.Markdown("""
    This app classifies goals into three categories:
    - **Vague**: Goals that lack specificity and measurable outcomes
    - **Partial SMART**: Goals that have some SMART elements but are missing others
    - **SMART**: Goals that are Specific, Measurable, Achievable, Relevant, and Time-bound
    
    Enter your goal below to see how it's classified and get feedback on how to improve it.
    """)
    
    with gr.Row():
        with gr.Column():
            goal_input = gr.Textbox(
                label="Enter your goal",
                placeholder="e.g., Complete a marathon in under 4 hours by following a 16-week training plan",
                lines=3
            )
            classify_button = gr.Button("Classify Goal", variant="primary")
        
        with gr.Column():
            classification_output = gr.Textbox(label="Classification Result")
            feedback_output = gr.Textbox(label="Feedback", lines=8)
    
    # Example goals
    gr.Examples(
        examples=[
            ["Become a better person"],
            ["Learn to play the guitar by practicing 30 minutes daily"],
            ["Complete a marathon in under 4 hours by following a 16-week training plan and running 4 times per week"],
            ["Improve my coding skills"],
            ["Read 2 books per month"],
            ["Lose 15 pounds by June by following a calorie-deficit diet and exercising 5 times a week, tracking progress weekly"]
        ],
        inputs=goal_input
    )
    
    # Set up the event handler
    classify_button.click(
        fn=classify_goal,
        inputs=goal_input,
        outputs=[classification_output, feedback_output]
    )
    
    # Also trigger classification when pressing Enter in the text box
    goal_input.submit(
        fn=classify_goal,
        inputs=goal_input,
        outputs=[classification_output, feedback_output]
    )

# Launch the app
demo.launch()
