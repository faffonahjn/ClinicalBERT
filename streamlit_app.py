# streamlit_app.py
import streamlit as st
import requests
import plotly.graph_objects as go

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title = 'ClinicalBERT Triage Classifier',
    page_icon  = '🏥',
    layout     = 'wide'
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5d6d7e;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #1a5276;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #5d6d7e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a5276;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
    }
    .prediction-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .confidence-value {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.2rem;
    }
    .sample-note {
        background: #fdfefe;
        border: 1px solid #d5d8dc;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #2c3e50;
        cursor: pointer;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        background: #1a5276;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: #2980b9;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #d5d8dc;
        text-align: center;
        color: #5d6d7e;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
import os
API_URL = os.environ.get('API_URL', 'http://127.0.0.1:8000') + '/predict'

TRIAGE_COLORS = {
    'Cardiovascular & Respiratory'  : '#e74c3c',
    'Emergency & Critical'          : '#c0392b',
    'Maternal, Child & Reproductive': '#8e44ad',
    'Medical & Primary Care'        : '#2980b9',
    'Neurological & Psychiatric'    : '#16a085',
    'Surgical'                      : '#e67e22',
}

TRIAGE_ICONS = {
    'Cardiovascular & Respiratory'  : '❤️',
    'Emergency & Critical'          : '🚨',
    'Maternal, Child & Reproductive': '👶',
    'Medical & Primary Care'        : '🩺',
    'Neurological & Psychiatric'    : '🧠',
    'Surgical'                      : '🔬',
}

SAMPLE_NOTES = {
    '🧠 Neurological' : (
        'Patient presents with sudden onset severe headache, '
        'photophobia, and neck stiffness. CT scan shows subarachnoid '
        'hemorrhage. Neurology consulted for management of intracranial pressure.'
    ),
    '❤️ Cardiovascular': (
        'Patient admitted with chest pain and shortness of breath. '
        'ECG shows ST elevation in leads V1 to V4. Troponin elevated at 2.4. '
        'Cardiology consulted for urgent cardiac catheterization and stent placement.'
    ),
    '🔬 Surgical'      : (
        'Patient is a 45 year old male presenting with right lower quadrant '
        'pain, rebound tenderness and guarding. WBC elevated at 14.5. '
        'CT abdomen confirms acute appendicitis. General surgery consulted '
        'for emergent laparoscopic appendectomy.'
    ),
    '👶 Maternal'      : (
        'Patient is a 28 year old female at 32 weeks gestation presenting '
        'with elevated blood pressure 160/110, severe headache and proteinuria. '
        'Diagnosis of severe preeclampsia. MFM consulted for delivery planning.'
    ),
    '🚨 Emergency'     : (
        'Patient brought in by EMS unresponsive after motor vehicle accident. '
        'GCS 6. Multiple rib fractures on chest xray. Massive hemothorax left side. '
        'Trauma surgery activated. Patient intubated in emergency bay.'
    ),
}

# ── HELPER FUNCTIONS ──────────────────────────────────────────
def call_predict_api(text: str) -> dict:
    """Call the FastAPI prediction endpoint."""
    response = requests.post(
        API_URL,
        json    = {'text': text},
        timeout = 60
    )
    response.raise_for_status()
    return response.json()


def build_confidence_chart(all_scores: dict) -> go.Figure:
    """Build a horizontal bar chart of class confidence scores."""
    categories = list(all_scores.keys())
    scores     = [all_scores[c] * 100 for c in categories]
    colors     = [TRIAGE_COLORS.get(c, '#2980b9') for c in categories]

    fig = go.Figure(go.Bar(
        x           = scores,
        y           = categories,
        orientation = 'h',
        marker_color= colors,
        text        = [f'{s:.1f}%' for s in scores],
        textposition= 'outside',
    ))

    fig.update_layout(
        title      = None,
        xaxis_title= 'Confidence (%)',
        yaxis_title= None,
        xaxis      = dict(range=[0, 110]),
        height     = 320,
        margin     = dict(l=10, r=60, t=10, b=30),
        plot_bgcolor   = 'rgba(0,0,0,0)',
        paper_bgcolor  = 'rgba(0,0,0,0)',
        font           = dict(size=12),
        showlegend     = False,
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🏥 ClinicalBERT')
    st.markdown('**Clinical Triage Classifier**')
    st.divider()

    # Model metrics
    st.markdown('### 📊 Model Performance')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro AUC</div>
            <div class="metric-value">0.879</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">6</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1</div>
            <div class="metric-value">0.474</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">4,966</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Sample notes
    st.markdown('### 📋 Sample Clinical Notes')
    st.markdown('Click a sample to load it:')

    for label, note in SAMPLE_NOTES.items():
        if st.button(label, key=f'sample_{label}'):
            st.session_state['input_text'] = note

    st.divider()

    # Model info
    st.markdown('### 🔧 Model Architecture')
    st.markdown("""
    - **Base**: Bio_ClinicalBERT
    - **Head**: Linear(768→256→6)
    - **Params**: 108.5M
    - **Training**: 5 epochs
    - **Optimizer**: AdamW (lr=2e-5)
    """)

    st.divider()
    st.markdown('### 🛠️ Tech Stack')
    st.markdown("""
    `PyTorch` · `HuggingFace` · `FastAPI`
    `Streamlit` · `Docker` · `Azure`
    """)


# ── MAIN PANEL ────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🏥 ClinicalBERT Triage Classifier</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">Fine-tuned Bio_ClinicalBERT · '
    '6-class clinical triage classification · '
    'Macro AUC 0.879</div>',
    unsafe_allow_html=True
)

# Input area
st.markdown('### 📝 Clinical Note Input')

input_text = st.text_area(
    label      = 'Enter clinical transcription text:',
    value      = st.session_state.get('input_text', ''),
    height     = 180,
    placeholder= (
        'Paste a clinical note here or select a sample from the sidebar...\n\n'
        'Example: Patient presents with chest pain, shortness of breath, '
        'and diaphoresis. ECG shows ST elevation...'
    ),
    key='input_text'
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    predict_clicked = st.button('🔍 Classify Triage Category', key='predict')
with col_clear:
    if st.button('🗑️ Clear', key='clear'):
        st.session_state['input_text'] = ''
        st.rerun()

# ── RESULTS ───────────────────────────────────────────────────
if predict_clicked:
    if not input_text or len(input_text.strip()) < 10:
        st.warning('⚠️ Please enter at least 10 characters of clinical text.')
    else:
        with st.spinner('🔄 Analysing clinical note...'):
            try:
                result = call_predict_api(input_text)

                category   = result['predicted_category']
                confidence = result['confidence']
                all_scores = result['all_scores']
                icon       = TRIAGE_ICONS.get(category, '🏥')
                color      = TRIAGE_COLORS.get(category, '#2980b9')

                st.markdown('---')
                st.markdown('### 🎯 Triage Classification Result')

                # Prediction card
                st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, {color}, #2c3e50);">
                    <div class="prediction-label">Predicted Triage Category</div>
                    <div class="prediction-value">{icon} {category}</div>
                    <div class="confidence-value">Confidence: {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence chart + score table
                col_chart, col_table = st.columns([3, 2])

                with col_chart:
                    st.markdown('#### Confidence Scores — All Classes')
                    fig = build_confidence_chart(all_scores)
                    st.plotly_chart(fig, use_container_width=True)

                with col_table:
                    st.markdown('#### Score Breakdown')
                    for cls, score in sorted(
                        all_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        icon_cls = TRIAGE_ICONS.get(cls, '•')
                        bar_pct  = int(score * 100)
                        is_top   = cls == category
                        weight   = '**' if is_top else ''
                        st.markdown(
                            f'{weight}{icon_cls} {cls}{weight}: `{score*100:.1f}%`'
                        )

                # Clinical note — what was analysed
                with st.expander('📄 View Analysed Text'):
                    st.text(input_text[:500] + ('...' if len(input_text) > 500 else ''))

            except requests.exceptions.ConnectionError:
                st.error(
                    '❌ Cannot connect to the prediction API. '
                    'Make sure the FastAPI server is running on port 8000.'
                )
            except Exception as e:
                st.error(f'❌ Prediction failed: {str(e)}')

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <strong>Francis Affonah</strong> · 
    Bio_ClinicalBERT fine-tuned on MTSamples (4,966 clinical notes) ·
    PyTorch · HuggingFace · FastAPI · Streamlit · Docker · Azure
</div>
""", unsafe_allow_html=True)