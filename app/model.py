# app/model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json

# Paths
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'clinicalbert_best.pt'
LABEL_PATH = BASE_DIR / 'processed' / 'label_map.json'

# Device — CPU for inference
device = torch.device('cpu')


class ClinicalBERTClassifier(nn.Module):
    """
    Bio_ClinicalBERT fine-tuned for 6-class clinical triage classification.
    """
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super(ClinicalBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


def load_label_map() -> dict:
    """Load integer → category name mapping."""
    with open(LABEL_PATH, 'r') as f:
        return json.load(f)


def load_model():
    """
    Load tokenizer and best model checkpoint from disk.
    Returns (model, tokenizer, label_map).
    Called once at API startup.
    """
    print(f'⬇️  Loading tokenizer...')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f'⬇️  Loading model weights...')
    model = ClinicalBERTClassifier(model_name, num_classes, dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    label_map = load_label_map()

    print(f'✅ Model loaded — {num_classes} classes')
    print(f'✅ Device: {device}')
    return model, tokenizer, label_map