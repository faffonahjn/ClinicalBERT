# app/main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schemas import PredictionRequest, PredictionResponse
from app.model import load_model
from app.predictor import predict
import torch

# Global model state
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, clean up at shutdown."""
    print('🚀 Starting ClinicalBERT Triage API...')
    ml_models['model'], ml_models['tokenizer'], ml_models['label_map'] = load_model()
    print('✅ API ready.')
    yield
    ml_models.clear()
    print('🛑 API shutdown.')

app = FastAPI(
    title       = 'ClinicalBERT Triage Classifier API',
    description = (
        'Fine-tuned Bio_ClinicalBERT for 6-class clinical triage classification. '
        'Classifies clinical transcription text into triage categories.'
    ),
    version  = '1.0.0',
    lifespan = lifespan
)


@app.get('/')
def root():
    """Health check endpoint."""
    return {
        'status'     : 'healthy',
        'model'      : 'Bio_ClinicalBERT',
        'task'       : '6-class clinical triage classification',
        'version'    : '1.0.0'
    }


@app.get('/health')
def health():
    """Detailed health check."""
    model_loaded = 'model' in ml_models
    return {
        'status'       : 'healthy' if model_loaded else 'loading',
        'model_loaded' : model_loaded,
        'device'       : str(torch.device('cpu'))
    }


@app.post('/predict', response_model=PredictionResponse)
def predict_triage(request: PredictionRequest):
    """
    Classify clinical text into one of 6 triage categories.

    Categories:
    - Cardiovascular & Respiratory
    - Emergency & Critical
    - Maternal, Child & Reproductive
    - Medical & Primary Care
    - Neurological & Psychiatric
    - Surgical
    """
    if 'model' not in ml_models:
        raise HTTPException(status_code=503, detail='Model not loaded yet.')

    try:
        predicted_category, confidence, all_scores = predict(
            text      = request.text,
            model     = ml_models['model'],
            tokenizer = ml_models['tokenizer'],
            label_map = ml_models['label_map'],
            device    = torch.device('cpu')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        predicted_category = predicted_category,
        confidence         = confidence,
        all_scores         = all_scores
    )