import streamlit as st
import torch
import os
from transformers import AutoTokenizer
import torch.nn.functional as F
from model import DistilBERTClass

# ------------------------------
# Load Model & Tokenizer
# ------------------------------
MODEL_PATH = r"PATH_TO_MODEL.pt" 
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
THRESHOLD = 0.32  # Minimum probability for a label to be displayed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBERTClass(num_labels=10)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Labels
LABELS = ['praise', 'amusement', 'anger', 'disapproval', 'confusion', 'interest', 'sadness', 'fear', 'joy', 'love']

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔍 Multi-Label Emotion Classifier")
st.markdown("Enter a sentence to classify its emotions.")

# User input
text_input = st.text_area("Enter text:", "")

if st.button("Classify"):
    if text_input.strip():
        # Tokenize input
        inputs = tokenizer(text_input, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Model inference
        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        # Apply sigmoid activation for multi-label classification
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Filter labels above threshold
        predicted_labels = [(label, prob) for label, prob in zip(LABELS, probabilities) if prob > THRESHOLD]
        predicted_labels = sorted(predicted_labels, key=lambda x: x[1], reverse=True)

        # Find highest scoring label that DID NOT pass the threshold
        below_threshold = [(label, prob) for label, prob in zip(LABELS, probabilities) if prob <= THRESHOLD]
        below_threshold = sorted(below_threshold, key=lambda x: x[1], reverse=True)

        # Display results
        if predicted_labels:
            st.subheader("Detected Emotions:")
            st.write(", ".join([label for label, _ in predicted_labels]))

        if below_threshold:
            st.subheader("Closest Missed Emotion:")
            st.write(f"**{below_threshold[0][0]}** (score: {below_threshold[0][1]:.3f})")

    else:
        st.warning("Please enter some text for classification.")
