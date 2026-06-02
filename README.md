# 🏥 ClinicalBERT Triage Classifier

Fine-tuned **Bio_ClinicalBERT** for 6-class clinical triage classification on real-world clinical transcription notes. Built with a custom PyTorch training pipeline and deployed via FastAPI, Streamlit, and Docker.

---

## 🎯 Project Overview

Clinical triage — the process of prioritizing patients by urgency — is one of the most critical workflows in healthcare.

This project fine-tunes a domain-specific BERT model (pre-trained on MIMIC-III clinical notes) to automatically classify clinical transcription text into one of six triage categories.

The project demonstrates an end-to-end machine learning engineering workflow:

**Data Pipeline → Transformer Fine-Tuning → Evaluation → Production API → Containerized Deployment**

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| **Macro AUC** | **0.879** |
| Macro F1 | 0.474 |
| Weighted F1 | 0.498 |
| Test Accuracy | 49.66% |
| Best Epoch | 5 of 8 |

### Per-Class AUC

| Triage Category | AUC |
|-------------------------------|--------|
| Neurological & Psychiatric | 0.9437 |
| Cardiovascular & Respiratory | 0.9046 |
| Maternal, Child & Reproductive | 0.9041 |
| Emergency & Critical | 0.8710 |
| Surgical | 0.8449 |
| Medical & Primary Care | 0.8050 |

---

## 🏗️ Architecture

```text
Clinical Text Input
        ↓
Bio_ClinicalBERT Encoder (12 Transformer Layers, 768 Hidden Units)
        ↓
CLS Token Extraction [batch_size, 768]
        ↓
Dropout(0.3)
        ↓
Linear(768 → 256)
        ↓
ReLU
        ↓
Dropout(0.3)
        ↓
Linear(256 → 6)
        ↓
Logits
        ↓
Softmax
        ↓
Predicted Triage Category + Confidence Score
```

### Model Details

| Component | Value |
|------------|---------|
| Base Model | Bio_ClinicalBERT |
| Hugging Face Model | `emilyalsentzer/Bio_ClinicalBERT` |
| Parameters | 108,508,678 |
| Trainable Parameters | 108,508,678 |
| Classification Head | Custom 2-Layer MLP |

---

## 🗂️ Triage Categories

| Label | Category |
|---------|--------------------------------|
| 0 | Cardiovascular & Respiratory |
| 1 | Emergency & Critical |
| 2 | Maternal, Child & Reproductive |
| 3 | Medical & Primary Care |
| 4 | Neurological & Psychiatric |
| 5 | Surgical |

---

## 📁 Project Structure

```text
ClinicalBERT/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── predictor.py
│   ├── schemas.py
│   └── __init__.py
│
├── logs/
│   ├── test_results.json
│   └── training_history.json
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── streamlit_app.py
├── requirements.txt
└── README.md
```

### Structure Overview

| File | Purpose |
|--------|---------|
| `main.py` | FastAPI application and endpoints |
| `model.py` | Model architecture and checkpoint loading |
| `predictor.py` | Inference pipeline |
| `schemas.py` | Pydantic request and response models |
| `streamlit_app.py` | Interactive web interface |
| `docker-compose.yml` | Multi-container orchestration |

---

## 🔧 Training Configuration

| Parameter | Value |
|------------|---------|
| Base Model | Bio_ClinicalBERT |
| Dataset | MTSamples |
| Total Samples | 4,966 |
| Train Set | 3,476 |
| Validation Set | 745 |
| Test Set | 745 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Loss Function | Weighted CrossEntropyLoss |
| Scheduler | Linear Warmup + Linear Decay |
| Warmup Ratio | 10% |
| Gradient Clipping | 1.0 |
| Batch Size | 16 |
| Max Sequence Length | 512 |
| Epochs | 8 |
| Best Epoch | 5 |
| Training Hardware | Tesla T4 GPU |

---

## 🚀 Running Locally

### Prerequisites

- Docker Desktop
- Python 3.11+
- Git

---

### Clone Repository

```bash
git clone https://github.com/faffonahjn/ClinicalBERT.git

cd ClinicalBERT
```

---

### Add Model Checkpoint

Download the trained model checkpoint and place it inside:

```text
models/
└── clinicalbert_best.pt
```

---

### Run With Docker

```bash
docker-compose up --build
```

---

### Services

| Service | URL |
|----------|---------|
| FastAPI Docs | http://localhost:8000/docs |
| FastAPI Health Check | http://localhost:8000/health |
| Streamlit UI | http://localhost:8501 |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|------------|------------|----------------|
| `/` | GET | Basic health check |
| `/health` | GET | Detailed health status |
| `/predict` | POST | Predict triage category |

---

## 📝 Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
"text": "Patient presents with sudden onset severe headache, photophobia and neck stiffness. CT scan shows subarachnoid hemorrhage."
}'
```

---

## 📤 Sample Response

```json
{
  "predicted_category": "Neurological & Psychiatric",
  "confidence": 0.7494,
  "all_scores": {
    "Neurological & Psychiatric": 0.7494,
    "Medical & Primary Care": 0.1248,
    "Emergency & Critical": 0.0400,
    "Surgical": 0.0370,
    "Maternal, Child & Reproductive": 0.0286,
    "Cardiovascular & Respiratory": 0.0203
  },
  "model_version": "Bio_ClinicalBERT-v1"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|----------|------------------------------|
| Deep Learning | PyTorch |
| Transformers | Hugging Face Transformers |
| Base Model | Bio_ClinicalBERT |
| API | FastAPI |
| ASGI Server | Uvicorn |
| Frontend | Streamlit |
| Visualization | Plotly |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Cloud Ready | Azure Container Apps |
| Training Environment | Google Colab |

---

## 📈 Training History

The model was trained for **8 epochs**, with the best checkpoint selected from **Epoch 5** based on validation loss.

To address class imbalance, weighted CrossEntropyLoss was used.

### Example Class Weights

| Category | Weight |
|-----------------------------|--------|
| Emergency & Critical | 4.42 |
| Medical & Primary Care | 0.40 |

This weighting strategy improved minority-class learning while maintaining overall predictive performance.

---

## 📄 Dataset

### MTSamples Medical Transcriptions Dataset

- 4,999 real-world medical transcription notes
- 40 clinical specialties
- Re-mapped into 6 triage categories
- Used for supervised clinical text classification

**Source:**  
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

---

## 👤 Author

### Francis Affonah

Registered Nurse → Clinical Data Scientist → Machine Learning Engineer

#### Connect

- GitHub: https://github.com/faffonahjn
- LinkedIn: https://linkedin.com/in/francis-affonah-23745a205
- Email: faffonahjnr@gmail.com

---

## 🌟 Key Highlights

✅ Fine-Tuned Bio_ClinicalBERT

✅ Clinical NLP Classification

✅ Custom PyTorch Training Pipeline

✅ Class-Imbalance Handling

✅ FastAPI Production API

✅ Streamlit Interactive Interface

✅ Dockerized Deployment

✅ Cloud-Ready Architecture

✅ Healthcare AI Portfolio Project

---

## 📜 License

This project is intended for educational, research, and portfolio purposes.

Not intended for clinical decision-making without further validation and regulatory approval.

---

### Built to demonstrate production-grade Healthcare NLP, Transformer Fine-Tuning, and MLOps Engineering.
