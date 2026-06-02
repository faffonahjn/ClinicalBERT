# app/predictor.py
import torch
import torch.nn.functional as F
import re
from typing import Dict, Tuple


def clean_text(text: str) -> str:
    """Apply same cleaning as training pipeline."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict(
    text       : str,
    model      ,
    tokenizer  ,
    label_map  : dict,
    max_length : int = 512,
    device     = torch.device('cpu')
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run inference on a single clinical text input.

    Returns:
        predicted_category : str
        confidence         : float
        all_scores         : dict of class → probability
    """
    # Clean text — same pipeline as training
    cleaned = clean_text(text)

    # Tokenize
    encoding = tokenizer(
        cleaned,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    # Softmax probabilities
    probs = F.softmax(logits, dim=1).squeeze(0)

    # Predicted class
    pred_idx  = torch.argmax(probs).item()
    pred_label = label_map[str(pred_idx)]
    confidence = probs[pred_idx].item()

    # All class scores
    all_scores = {
        label_map[str(i)]: round(probs[i].item(), 4)
        for i in range(len(label_map))
    }

    return pred_label, round(confidence, 4), all_scores