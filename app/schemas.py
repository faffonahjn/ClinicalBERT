# app/schemas.py
from pydantic import BaseModel, Field
from typing import Dict

class PredictionRequest(BaseModel):
    """Incoming request — clinical text to classify."""
    text: str = Field(
        ...,
        min_length=10,
        description="Clinical transcription text to classify"
    )

class TriagePrediction(BaseModel):
    """Single triage prediction with confidence scores."""
    predicted_category : str
    confidence         : float
    all_scores         : Dict[str, float]

class PredictionResponse(BaseModel):
    """Full API response."""
    predicted_category : str
    confidence         : float
    all_scores         : Dict[str, float]
    model_version      : str = "Bio_ClinicalBERT-v1"