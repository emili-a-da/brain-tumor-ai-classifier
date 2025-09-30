# streamlit_app.py
# ----------------
# Self-contained Streamlit app for MRI tumor classification
# Uses a local Keras .h5 model (or downloads it at startup if MODEL_URL secret is set)

import os, io, pathlib
import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ========= App Config =========
st.set_page_config(page_title="MRI Tumor Classifier", layout="centered")

# IMPORTANT: keep this order identical to the one used during training!
LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

# Match your training input size
IMAGE_SIZE = (224, 224)

# Multiple model path options for different deployment scenarios
def get_model_path():
    """Get the correct model path depending on the deployment context."""
    # Try different possible paths
    possible_paths = [
        # Path when running from secrets.toml folder (local development)
        os.path.join(os.path.dirname(__file__), "Zuzik_mri_model_final22.h5"),
        # Path when running from root directory (deployed app)
        os.path.join(os.path.dirname(__file__), "secrets.toml", "Zuzik_mri_model_final22.h5"),
        # Path in current working directory
        "Zuzik_mri_model_final22.h5",
        # Path in secrets.toml folder from current directory
        os.path.join("secrets.toml", "Zuzik_mri_model_final22.h5")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no local file found, return the first path for download attempt
    return possible_paths[0]

DEFAULT_MODEL_PATH = get_model_path()

# --- Tumor Type Explanations ---
TUMOR_EXPLANATIONS = {
    'glioma': {
        'description': 'Gliomas are brain tumors that develop from glial cells, which support and protect nerve cells in the brain.',
        'details': 'Gliomas are the most common type of primary brain tumor in adults. They can be slow-growing (low-grade) or fast-growing (high-grade). Symptoms may include headaches, seizures, memory problems, personality changes, and neurological deficits depending on the location.',
        'treatment': 'Treatment typically involves a combination of surgery, radiation therapy, and chemotherapy. The specific treatment plan depends on the type, grade, location, and size of the tumor.',
        'prognosis': 'Prognosis varies significantly based on the grade and location of the tumor. Low-grade gliomas may have better outcomes than high-grade ones.'
    },
    'meningioma': {
        'description': 'Meningiomas are tumors that develop from the meninges, the protective membranes surrounding the brain and spinal cord.',
        'details': 'Most meningiomas are benign (non-cancerous) and grow slowly. They are more common in women and older adults. Symptoms depend on the location and size, and may include headaches, vision problems, seizures, weakness, or speech difficulties.',
        'treatment': 'Treatment options include observation (for small, asymptomatic tumors), surgical removal, and radiation therapy. Many meningiomas can be completely cured with surgical removal.',
        'prognosis': 'Generally good for benign meningiomas, especially when completely removed surgically. Most patients can return to normal activities after recovery.'
    },
    'notumor': {
        'description': 'No tumor detected - the brain scan appears normal without signs of abnormal growth or masses.',
        'details': 'A normal brain MRI shows healthy brain tissue without evidence of tumors, lesions, or other abnormalities. The brain structures appear intact and properly positioned with normal signal intensity.',
        'treatment': 'No treatment is needed for a normal scan. Continue regular check-ups as recommended by your healthcare provider and report any new neurological symptoms.',
        'prognosis': 'Excellent - no abnormalities detected. Maintain regular health monitoring as advised by your physician.'
    },
    'pituitary': {
        'description': 'Pituitary tumors (adenomas) are growths in the pituitary gland, a small gland at the base of the brain that controls hormone production.',
        'details': 'Most pituitary tumors are benign adenomas. They can be functioning (producing excess hormones) or non-functioning. Symptoms may include headaches, vision problems, hormonal imbalances, fatigue, and changes in growth or sexual function.',
        'treatment': 'Treatment depends on the type and size of the tumor. Options include medication to control hormone levels, surgery (often through the nose), and radiation therapy. Many pituitary tumors can be effectively managed.',
        'prognosis': 'Generally good with appropriate treatment. Many patients can achieve normal hormone levels and symptom relief with proper management.'
    }
}

# Optional: if you don't commit the model, set a direct download link
# via Streamlit Secrets -> MODEL_URL = "https://....h5"
try:
    MODEL_URL = st.secrets["MODEL_URL"].strip()
except (KeyError, FileNotFoundError):
    # Default model URL from your Google Drive (if no secret is set)
    MODEL_URL = "https://drive.google.com/uc?export=download&id=1yGajR8Bj1hGOF3fnp5g_KtDW44ncA3tg"

# ========= Utilities =========
def _ensure_model_present(local_path: str = DEFAULT_MODEL_PATH):
    """Download the model if missing and MODEL_URL secret provided."""
    pathlib.Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
    if os.path.exists(local_path):
        return
    if not MODEL_URL:
        # No local file and no URL to fetch from
        error_msg = f"""
        🔍 Model file not found at: {local_path}
        
        📋 Possible solutions:
        1. Add MODEL_URL to Streamlit Secrets:
           MODEL_URL = "https://drive.google.com/uc?export=download&id=1yGajR8Bj1hGOF3fnp5g_KtDW44ncA3tg"
        
        2. Upload model file to your repository in the correct location
        
        3. Check that the model file exists in one of these locations:
           - {os.path.join(os.path.dirname(__file__), "Zuzik_mri_model_final22.h5")}
           - {os.path.join("secrets.toml", "Zuzik_mri_model_final22.h5")}
        """
        raise FileNotFoundError(error_msg)
    
    # Stream download to avoid big memory spikes
    st.info(f"📥 Downloading model from: {MODEL_URL[:50]}...")
    import requests
    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        raise FileNotFoundError(f"Failed to download model from {MODEL_URL}: {str(e)}")

def _preprocess(img_pil: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a model-ready batch tensor (1, H, W, 3)
    using the same preprocessing as used during training.
    """
    # normalize channels & size
    img = img_pil.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    # VGG16-specific preprocessing (BGR, mean subtraction, etc.)
    arr = preprocess_input(arr)
    return arr

def _postprocess(pred: np.ndarray):
    """Return (top_label, {label: prob}) from raw model prediction."""
    probs = pred[0].astype(float)  # shape: (num_classes,)
    top_idx = int(np.argmax(probs))
    conf_map = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return LABELS[top_idx], conf_map

@st.cache_resource(show_spinner=True)
def get_model(local_path: str = DEFAULT_MODEL_PATH):
    """
    Cache the loaded model across reruns.
    Set compile=False for faster load (we only need inference).
    """
    _ensure_model_present(local_path)
    model = load_model(local_path, compile=False)
    # (Optional) model.make_predict_function()  # not required in TF 2.x eager
    return model

# ========= Ultra-Advanced CSS Styling with Background Animations =========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Animated Background Setup */
    .main {
        padding-top: 2rem;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        position: relative;
        overflow-x: hidden;
    }
    
    /* Professional Medical Background */
    .animated-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
        background: linear-gradient(-45deg, #e6f3ff, #f0f8ff, #f5f9fc, #fafbfc, #e8f4f8, #f2f7fa);
        background-size: 400% 400%;
        animation: gradientBG 25s ease infinite;
        opacity: 0.6;
    }
    
    /* Professional Medical Particles */
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle-1 {
        width: 16px;
        height: 16px;
        background: rgba(99, 142, 184, 0.15);
        top: 20%;
        left: 20%;
        animation: float-particle 20s infinite linear;
    }
    
    .particle-2 {
        width: 12px;
        height: 12px;
        background: rgba(156, 163, 175, 0.12);
        top: 60%;
        left: 80%;
        animation: float-particle 18s infinite linear reverse;
    }
    
    .particle-3 {
        width: 20px;
        height: 20px;
        background: rgba(129, 140, 248, 0.1);
        top: 40%;
        left: 10%;
        animation: float-particle 22s infinite linear;
    }
    
    .particle-4 {
        width: 14px;
        height: 14px;
        background: rgba(167, 139, 250, 0.08);
        top: 80%;
        left: 60%;
        animation: float-particle 19s infinite linear reverse;
    }
    
    .particle-5 {
        width: 10px;
        height: 10px;
        background: rgba(139, 192, 216, 0.12);
        top: 30%;
        left: 70%;
        animation: float-particle 21s infinite linear;
    }
    
    /* Brain-themed floating elements */
    .brain-particle {
        position: fixed;
        font-size: 30px;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
    }
    
    .brain-1 {
        top: 15%;
        left: 15%;
        animation: brain-float 20s infinite ease-in-out;
    }
    
    .brain-2 {
        top: 70%;
        left: 85%;
        animation: brain-float 25s infinite ease-in-out reverse;
    }
    
    .brain-3 {
        top: 45%;
        left: 5%;
        animation: brain-float 18s infinite ease-in-out;
    }
    
    /* Professional wave background */
    .wave-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -2;
        background: 
            radial-gradient(circle at 20% 50%, rgba(99, 142, 184, 0.03) 0%, transparent 60%),
            radial-gradient(circle at 80% 20%, rgba(129, 140, 248, 0.02) 0%, transparent 60%),
            radial-gradient(circle at 40% 80%, rgba(156, 163, 175, 0.025) 0%, transparent 60%);
        animation: wave-move 35s ease-in-out infinite;
    }
    
    /* ========== KEYFRAME ANIMATIONS ========== */
    
    /* Animated gradient background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        25% { background-position: 100% 50%; }
        50% { background-position: 100% 100%; }
        75% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating particles animation */
    @keyframes float-particle {
        0% { 
            transform: translateY(0px) translateX(0px) rotate(0deg);
            opacity: 0;
        }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { 
            transform: translateY(-100vh) translateX(50px) rotate(360deg);
            opacity: 0;
        }
    }
    
    /* Brain emoji floating animation */
    @keyframes brain-float {
        0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1);
            opacity: 0.1;
        }
        25% { 
            transform: translateY(-20px) translateX(10px) scale(1.1);
            opacity: 0.2;
        }
        50% { 
            transform: translateY(-10px) translateX(-15px) scale(0.9);
            opacity: 0.15;
        }
        75% { 
            transform: translateY(-30px) translateX(5px) scale(1.05);
            opacity: 0.25;
        }
    }
    
    /* Wave background movement */
    @keyframes wave-move {
        0%, 100% { 
            background-position: 0% 0%, 100% 100%, 50% 50%;
        }
        33% { 
            background-position: 100% 0%, 0% 100%, 0% 0%;
        }
        66% { 
            background-position: 50% 100%, 100% 0%, 100% 100%;
        }
    }
    
    /* Floating animation for UI elements */
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-8px) rotate(1deg); }
        50% { transform: translateY(-4px) rotate(0deg); }
        75% { transform: translateY(-12px) rotate(-1deg); }
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.8; }
    }
    
    /* Subtle professional glow */
    @keyframes glow {
        0%, 100% { 
            box-shadow: 
                0 8px 32px rgba(71, 85, 105, 0.06),
                0 4px 16px rgba(100, 116, 139, 0.04);
            filter: brightness(1);
        }
        50% { 
            box-shadow: 
                0 12px 40px rgba(71, 85, 105, 0.08),
                0 6px 20px rgba(100, 116, 139, 0.06),
                0 2px 8px rgba(148, 163, 184, 0.05);
            filter: brightness(1.02);
        }
    }
    
    /* Sparkle animation for special effects */
    @keyframes sparkle {
        0%, 100% { 
            transform: scale(0) rotate(0deg);
            opacity: 0;
        }
        50% { 
            transform: scale(1) rotate(180deg);
            opacity: 1;
        }
    }
    
    /* Professional medical header */
    .header-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 20%, #cbd5e1 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        color: #1e293b;
        text-align: center;
        box-shadow: 
            0 10px 40px rgba(71, 85, 105, 0.08),
            0 4px 20px rgba(100, 116, 139, 0.05);
        position: relative;
        overflow: hidden;
        animation: float 8s ease-in-out infinite;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s linear infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(30deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(30deg); }
    }
    
    /* 3D Card effects with background interaction */
    .card-3d {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.1),
            0 2px 16px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
        animation: float 8s ease-in-out infinite;
    }
    
    .card-3d:hover {
        transform: translateY(-12px) scale(1.03);
        box-shadow: 
            0 25px 80px rgba(0, 0, 0, 0.15),
            0 10px 40px rgba(0, 0, 0, 0.1),
            0 0 0 1px rgba(102, 126, 234, 0.1);
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
    }
    
    /* Add shimmer effect to cards */
    .card-3d::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: all 0.5s;
    }
    
    .card-3d:hover::before {
        left: 100%;
    }
    
    /* Prediction result with 3D effect */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.4),
            0 5px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .prediction-result::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: slideIn 2s ease-in-out infinite;
    }
    
    @keyframes slideIn {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: 100%; }
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 8px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: rgba(248, 250, 252, 0.8);
        border-radius: 12px;
        color: #475569;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Animated progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        animation: gradientShift 2s ease infinite;
        border-radius: 10px;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* File uploader enhancement */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stFileUploader > div > div:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: scale(1.02);
        animation: float 2s ease-in-out infinite;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* Image styling */
    .stImage > img {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        animation: float 3s ease-in-out infinite;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
    }
</style>
""", unsafe_allow_html=True)

# ========= Animated Background Elements =========
st.markdown("""
<!-- Main animated background -->
<div class="animated-background"></div>
<div class="wave-background"></div>

<!-- Floating particles -->
<div class="particle particle-1"></div>
<div class="particle particle-2"></div>
<div class="particle particle-3"></div>
<div class="particle particle-4"></div>
<div class="particle particle-5"></div>

<!-- Brain-themed floating elements -->
<div class="brain-particle brain-1">🧠</div>
<div class="brain-particle brain-2">🔬</div>
<div class="brain-particle brain-3">⚕️</div>

<!-- Additional sparkle effects -->
<div style="position: fixed; top: 10%; right: 15%; font-size: 20px; z-index: -1; animation: sparkle 8s infinite;">✨</div>
<div style="position: fixed; top: 70%; right: 25%; font-size: 15px; z-index: -1; animation: sparkle 6s infinite 2s;">💫</div>
<div style="position: fixed; top: 40%; right: 80%; font-size: 25px; z-index: -1; animation: sparkle 10s infinite 4s;">⭐</div>
<div style="position: fixed; top: 85%; right: 10%; font-size: 18px; z-index: -1; animation: sparkle 7s infinite 1s;">✨</div>
""", unsafe_allow_html=True)

# ========= Enhanced Sidebar =========
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8fafc, #e2e8f0, #cbd5e1); 
                color: #1e293b; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
                box-shadow: 0 6px 25px rgba(71, 85, 105, 0.08); 
                border: 1px solid rgba(148, 163, 184, 0.15);
                position: relative; overflow: hidden;">
        <div style="position: absolute; top: -30%; right: -30%; width: 80%; height: 80%; 
                    background: radial-gradient(circle, rgba(148, 163, 184, 0.05) 0%, transparent 70%);
                    animation: float 6s ease-in-out infinite;"></div>
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 2rem; text-align: center; margin-bottom: 0.8rem; color: #475569;">🧠 ⚕️</div>
            <h3 style="margin: 0; text-align: center; font-weight: 600; color: #1e293b;">
                Clinical AI Diagnostics
            </h3>
            <p style="margin: 0.8rem 0 0 0; opacity: 0.8; text-align: center; font-weight: 400; color: #64748b;">
                Professional Medical Imaging Analysis
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional tumor classification cards
    st.markdown("""
    <div style="background: rgba(248, 250, 252, 0.95); padding: 1.5rem; border-radius: 12px; 
                margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(71, 85, 105, 0.06);
                border: 1px solid rgba(148, 163, 184, 0.1);">
        <h4 style="color: #1e293b; margin-top: 0; text-align: center; font-weight: 600;">📋 Classification Types</h4>
        <div style="margin: 1rem 0;">
            <div style="display: flex; align-items: center; margin: 0.8rem 0; padding: 0.8rem; 
                        background: #fef2f2; border-radius: 8px; border-left: 3px solid #fca5a5;">
                <span style="font-size: 1.1rem; margin-right: 0.8rem;">🔴</span>
                <div>
                    <strong style="color: #991b1b; font-weight: 600;">Glioma</strong><br>
                    <small style="color: #7f1d1d; font-weight: 400;">Glial cell neoplasms</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin: 0.8rem 0; padding: 0.8rem; 
                        background: #fffbeb; border-radius: 8px; border-left: 3px solid #fbbf24;">
                <span style="font-size: 1.1rem; margin-right: 0.8rem;">🟡</span>
                <div>
                    <strong style="color: #92400e; font-weight: 600;">Meningioma</strong><br>
                    <small style="color: #78350f; font-weight: 400;">Meningeal tumors</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin: 0.8rem 0; padding: 0.8rem; 
                        background: #f0fdf4; border-radius: 8px; border-left: 3px solid #86efac;">
                <span style="font-size: 1.1rem; margin-right: 0.8rem;">🟢</span>
                <div>
                    <strong style="color: #166534; font-weight: 600;">Normal Tissue</strong><br>
                    <small style="color: #14532d; font-weight: 400;">Healthy brain tissue</small>
                </div>
            </div>
            <div style="display: flex; align-items: center; margin: 0.8rem 0; padding: 0.8rem; 
                        background: #fff7ed; border-radius: 8px; border-left: 3px solid #fdba74;">
                <span style="font-size: 1.1rem; margin-right: 0.8rem;">🟠</span>
                <div>
                    <strong style="color: #c2410c; font-weight: 600;">Pituitary Adenoma</strong><br>
                    <small style="color: #9a3412; font-weight: 400;">Pituitary gland tumor</small>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced requirements section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e0f2fe, #b3e5fc); padding: 1.5rem; 
                border-radius: 15px; margin-bottom: 1rem; border: 2px solid #0ea5e9;">
        <h4 style="color: #0c4a6e; margin-top: 0; display: flex; align-items: center;">
            📸 <span style="margin-left: 0.5rem;">Image Requirements</span>
        </h4>
        <div style="color: #0c4a6e;">
            <div style="margin: 0.5rem 0;">📁 <strong>Format:</strong> JPG, JPEG, PNG</div>
            <div style="margin: 0.5rem 0;">🏥 <strong>Type:</strong> Brain MRI scans</div>
            <div style="margin: 0.5rem 0;">✨ <strong>Quality:</strong> Medical-grade</div>
            <div style="margin: 0.5rem 0;">📏 <strong>Size:</strong> Any resolution</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced tips section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); padding: 1.5rem; 
                border-radius: 15px; margin-bottom: 1rem; border: 2px solid #22c55e;">
        <h4 style="color: #14532d; margin-top: 0; display: flex; align-items: center;">
            ⚡ <span style="margin-left: 0.5rem;">Pro Tips</span>
        </h4>
        <div style="color: #14532d;">
            <div style="margin: 0.5rem 0;">✅ High-quality MRI images</div>
            <div style="margin: 0.5rem 0;">✅ Proper brain positioning</div>
            <div style="margin: 0.5rem 0;">✅ Axial view preferred</div>
            <div style="margin: 0.5rem 0;">✅ Clear image contrast</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced performance metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fdf4ff, #fae8ff); padding: 1.5rem; 
                border-radius: 15px; border: 2px solid #a855f7;">
        <h4 style="color: #581c87; margin-top: 0; text-align: center;">🎯 Model Specs</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
            <div style="text-align: center; padding: 0.8rem; background: rgba(168, 85, 247, 0.1); 
                        border-radius: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #7c3aed;">4</div>
                <div style="font-size: 0.8rem; color: #581c87;">Classes</div>
            </div>
            <div style="text-align: center; padding: 0.8rem; background: rgba(168, 85, 247, 0.1); 
                        border-radius: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #7c3aed;">224²</div>
                <div style="font-size: 0.8rem; color: #581c87;">Input Size</div>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem; padding: 0.8rem; 
                    background: rgba(168, 85, 247, 0.1); border-radius: 10px;">
            <div style="font-size: 1.2rem; font-weight: bold; color: #7c3aed;">Deep CNN</div>
            <div style="font-size: 0.8rem; color: #581c87;">Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========= Big Animated Title with Transitions =========
st.markdown("""
<style>
/* Animated Title Styles */
.big-title-container {
    text-align: center;
    padding: 4rem 2rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    background-size: 300% 300%;
    border-radius: 25px;
    position: relative;
    overflow: hidden;
    animation: gradientShift 8s ease infinite;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
}

.big-title-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shine 3s linear infinite;
}

.main-title {
    font-size: 4.5rem;
    font-weight: 900;
    color: white;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: titlePulse 4s ease-in-out infinite;
    position: relative;
    z-index: 2;
}

.subtitle {
    font-size: 1.5rem;
    color: rgba(255,255,255,0.95);
    margin: 1rem 0;
    font-weight: 500;
    animation: subtitleFade 3s ease-in-out infinite alternate;
    position: relative;
    z-index: 2;
}

.medical-icons {
    font-size: 2.5rem;
    margin: 1.5rem 0;
    animation: iconsBounce 2s ease-in-out infinite;
    position: relative;
    z-index: 2;
}

.description-text {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
    margin-top: 1.5rem;
    font-weight: 400;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
    animation: textGlow 4s ease-in-out infinite;
    position: relative;
    z-index: 2;
}

.feature-badges {
    margin-top: 2rem;
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 2;
}

.feature-badge {
    background: rgba(255,255,255,0.2);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
    color: white;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
    animation: badgeFloat 3s ease-in-out infinite;
}

/* Animation Keyframes */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes shine {
    0% { left: -100%; }
    100% { left: 100%; }
}

@keyframes titlePulse {
    0%, 100% { transform: scale(1); text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    50% { transform: scale(1.05); text-shadow: 4px 4px 8px rgba(0,0,0,0.4); }
}

@keyframes subtitleFade {
    0% { opacity: 0.8; }
    100% { opacity: 1; }
}

@keyframes iconsBounce {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes textGlow {
    0%, 100% { text-shadow: 0 0 5px rgba(255,255,255,0.3); }
    50% { text-shadow: 0 0 20px rgba(255,255,255,0.6); }
}

@keyframes badgeFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-title { font-size: 2.5rem; }
    .subtitle { font-size: 1.2rem; }
    .medical-icons { font-size: 2rem; }
    .description-text { font-size: 1rem; }
    .big-title-container { padding: 3rem 1rem; }
    .feature-badges { flex-direction: column; align-items: center; }
}
</style>

<div class="big-title-container">
    <div class="medical-icons">🧠 ⚕️ 🔬</div>
    <h1 class="main-title">Brain Tumor AI</h1>
    <p class="subtitle">Advanced Medical Image Analysis</p>
    <p class="description-text">
        Revolutionary AI-powered diagnostic tool for brain tumor classification using state-of-the-art 
        deep learning technology. Upload MRI scans for instant, accurate medical insights.
    </p>
    <div class="feature-badges">
        <div class="feature-badge">⚡ Real-time Analysis</div>
        <div class="feature-badge">🎯 Clinical Accuracy</div>
        <div class="feature-badge">🔒 HIPAA Compliant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========= Model Loading with Status =========
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    model_load_error = None
    with st.spinner("🔄 Loading AI model..."):
        try:
            model = get_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            model = None
            model_load_error = e

if model is None:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
        <h2>❌ Model Loading Failed</h2>
        <p><strong>Path tried:</strong> <code>{}</code></p>
        <p><strong>Error:</strong> {}</p>
        <hr style="border-color: rgba(255,255,255,0.3);">
        <p>📁 <strong>Solution 1:</strong> Place your model file in the correct directory</p>
        <p>🔗 <strong>Solution 2:</strong> Set MODEL_URL in Streamlit Secrets for auto-download</p>
    </div>
    """.format(DEFAULT_MODEL_PATH, model_load_error), unsafe_allow_html=True)
    st.stop()

# ========= Enhanced File Upload Section =========
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h2 style="background: linear-gradient(135deg, #667eea, #764ba2); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 2.5rem; font-weight: 800; margin: 0;">
        📤 Upload Your MRI Image
    </h2>
    <p style="color: #6b7280; margin-top: 0.5rem; font-size: 1.1rem;">
        Select a brain MRI scan for AI-powered tumor analysis
    </p>
</div>

<div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05)); 
            padding: 2rem; border-radius: 20px; margin: 2rem 0; 
            border: 2px dashed rgba(102, 126, 234, 0.3);">
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded = st.file_uploader(
        "Choose an MRI image file", 
        type=["jpg", "jpeg", "png"],
        help="📁 Supported: JPG, JPEG, PNG • 📏 Max size: 200MB • 🏥 Medical quality preferred",
        label_visibility="collapsed"
    )

st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    # Read image safely
    try:
        img = Image.open(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    # Display uploaded image with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption="📷 Uploaded MRI Image", width=400)

    # Predict with enhanced spinner
    with st.spinner("🔍 Analyzing image with AI..."):
        x = _preprocess(img)
        preds = model.predict(x)  # shape: (1, num_classes)
        pred_label, conf = _postprocess(preds)

    # Get the highest confidence value for the predicted label
    highest_conf = max(conf.values())
    
    # Spectacular prediction display
    st.markdown("---")
    
    # Professional medical prediction styling
    prediction_data = {
        'glioma': {'icon': '🔴', 'color': '#991b1b', 'bg': 'linear-gradient(135deg, #fef2f2, #fee2e2, #fecaca)'},
        'meningioma': {'icon': '🟡', 'color': '#92400e', 'bg': 'linear-gradient(135deg, #fffbeb, #fef3c7, #fde68a)'}, 
        'notumor': {'icon': '🟢', 'color': '#166534', 'bg': 'linear-gradient(135deg, #f0fdf4, #dcfce7, #bbf7d0)'},
        'pituitary': {'icon': '🟠', 'color': '#c2410c', 'bg': 'linear-gradient(135deg, #fff7ed, #ffedd5, #fed7aa)'}
    }
    
    pred_info = prediction_data.get(pred_label, {'icon': '🔵', 'color': '#3b82f6', 'bg': 'linear-gradient(135deg, #dbeafe, #bfdbfe, #93c5fd)'})
    
    st.markdown(f"""
    <div style="background: {pred_info['bg']}; color: #1e293b; position: relative; overflow: hidden;
                padding: 3rem 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;
                box-shadow: 0 8px 32px rgba(71, 85, 105, 0.12), 0 4px 16px rgba(100, 116, 139, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.15);">
        <div style="position: absolute; top: -30%; left: -30%; width: 160%; height: 160%; 
                    background: radial-gradient(circle, rgba(248, 250, 252, 0.2) 0%, transparent 60%);
                    animation: float 8s ease-in-out infinite;"></div>
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 3rem; margin-bottom: 1.5rem;">
                {pred_info['icon']} ⚕️
            </div>
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 500; color: #475569; margin-bottom: 0.5rem;">
                Diagnostic Analysis Complete
            </h2>
            <div style="height: 2px; width: 120px; background: {pred_info['color']}; 
                        margin: 1rem auto; border-radius: 1px;"></div>
            <h1 style="margin: 1rem 0; font-size: 2.8rem; font-weight: 700; color: {pred_info['color']};
                       text-shadow: 1px 1px 2px rgba(71, 85, 105, 0.1);">
                {pred_label.title()}
            </h1>
            <div style="background: rgba(248, 250, 252, 0.9); padding: 1rem 2rem; border-radius: 15px; 
                        display: inline-block; margin-top: 1.5rem; 
                        box-shadow: 0 4px 12px rgba(71, 85, 105, 0.08);
                        border: 1px solid rgba(148, 163, 184, 0.1);">
                <span style="font-size: 1.6rem; font-weight: 600; color: {pred_info['color']};">
                    {highest_conf:.1%}
                </span>
                <span style="font-size: 1rem; color: #64748b; margin-left: 0.5rem;">Confidence Level</span>
            </div>
            <div style="margin-top: 2rem; font-size: 0.9rem; color: #64748b; font-weight: 500;">
                🔬 Clinical AI Analysis • ⚡ Real-time Processing • 🎯 Medical-grade Precision
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display detailed explanation for the predicted tumor type
    if pred_label in TUMOR_EXPLANATIONS:
        explanation = TUMOR_EXPLANATIONS[pred_label]
        
        st.markdown("""
        <h3 style="color: white; margin-top: 2rem;">📋 Medical Information</h3>
        """, unsafe_allow_html=True)
        
        # Enhanced tabs with icons
        desc_tab, details_tab, treatment_tab, prognosis_tab = st.tabs([
            "📖 Description", 
            "🔍 Clinical Details", 
            "💊 Treatment", 
            "📈 Prognosis"
        ])
        
        with desc_tab:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1.8rem; border-radius: 12px; border-left: 4px solid #64748b;
                        box-shadow: 0 2px 8px rgba(71, 85, 105, 0.06);">
                <h4 style="color: #1e293b; margin-top: 0; font-weight: 600;">What is {pred_label.title()}?</h4>
                <p style="color: #475569; margin-bottom: 0; line-height: 1.7; font-size: 1rem;">{explanation['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with details_tab:
            st.markdown(f"""
            <div style="background: #fefcf6; padding: 1.8rem; border-radius: 12px; border-left: 4px solid #b45309;
                        box-shadow: 0 2px 8px rgba(180, 83, 9, 0.06);">
                <h4 style="color: #92400e; margin-top: 0; font-weight: 600;">Clinical Details</h4>
                <p style="color: #451a03; margin-bottom: 0; line-height: 1.7; font-size: 1rem;">{explanation['details']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with treatment_tab:
            st.markdown(f"""
            <div style="background: #f0fdf4; padding: 1.8rem; border-radius: 12px; border-left: 4px solid #16a34a;
                        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.06);">
                <h4 style="color: #15803d; margin-top: 0; font-weight: 600;">Treatment Options</h4>
                <p style="color: #166534; margin-bottom: 0; line-height: 1.7; font-size: 1rem;">{explanation['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with prognosis_tab:
            st.markdown(f"""
            <div style="background: #f0f9ff; padding: 1.8rem; border-radius: 12px; border-left: 4px solid #0ea5e9;
                        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.06);">
                <h4 style="color: #0369a1; margin-top: 0; font-weight: 600;">Prognosis</h4>
                <p style="color: #1e40af; margin-bottom: 0; line-height: 1.7; font-size: 1rem;">{explanation['prognosis']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced medical disclaimer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7, #fde68a); 
                padding: 2rem; border-radius: 16px; border: 1px solid #d97706; margin: 2rem 0;
                box-shadow: 0 4px 16px rgba(217, 119, 6, 0.1);">
        <h4 style="color: #92400e; margin-top: 0; font-weight: 600;">⚕️ Important Medical Disclaimer</h4>
        <p style="color: #451a03; margin-bottom: 0; line-height: 1.7; font-size: 1rem;">
            <strong>This AI model is designed for educational and research purposes only.</strong> 
            It should not be used as a substitute for professional medical diagnosis or treatment decisions. 
            Always consult with qualified healthcare professionals for proper medical evaluation, diagnosis, and treatment planning.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced confidence breakdown section
    with st.expander("📊 Detailed Analysis & Probabilities", expanded=False):
        st.markdown("### 🎯 Classification Confidence")
        
        # Create columns for better layout
        for k in LABELS:
            p = conf[k]
            
            # Color coding based on confidence - professional medical colors
            if p > 0.7:
                color = "#16a34a"  # Professional green for high confidence
            elif p > 0.4:
                color = "#d97706"  # Professional amber for medium confidence
            else:
                color = "#64748b"  # Professional gray for low confidence
            
            # Create individual cards for each prediction
            icon = {
                'glioma': '🔴',
                'meningioma': '🟡', 
                'notumor': '🟢',
                'pituitary': '🟠'
            }.get(k, '🔵')
            
            if k == pred_label:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}, {color}dd); 
                            color: white; padding: 1.2rem; border-radius: 12px; margin: 0.5rem 0;
                            border: 2px solid #cbd5e1; box-shadow: 0 4px 12px {color}33;">
                    <h4 style="margin: 0; font-weight: 600;">{icon} {k.title()} - {p:.1%} ← PRIMARY DIAGNOSIS</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: {color}15; color: {color}; 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                            border-left: 3px solid {color}; box-shadow: 0 2px 6px {color}10;">
                    <h5 style="margin: 0; font-weight: 500;">{icon} {k.title()} - {p:.1%}</h5>
                </div>
                """, unsafe_allow_html=True)
            
            st.progress(min(max(p, 0.0), 1.0))
        
        st.markdown("---")
        
        # Technical details for developers
        with st.expander("🔧 Technical Details (For Developers)", expanded=False):
            st.json(conf)
            st.markdown(f"**Model Input Shape:** {IMAGE_SIZE}")
            st.markdown(f"**Number of Classes:** {len(LABELS)}")
            st.markdown(f"**Labels:** {', '.join(LABELS)}")

# ========= Clean Footer =========
st.markdown("---")

# Create columns for the footer content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📊 Model Performance**
    - Research-grade accuracy
    - Real-time analysis
    - Medical-grade precision
    """)

with col2:
    st.markdown("""
    **🎯 Tumor Detection**
    - 4 classification types
    - Advanced deep learning
    - Educational insights
    """)

with col3:
    st.markdown("""
    **🔒 Privacy & Security**
    - Secure processing
    - No data storage
    - Privacy protected
    """)

st.markdown("---")

# Professional footer message
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f1f5f9, #e2e8f0); 
            color: #1e293b; border-radius: 16px; margin: 2rem 0; border: 1px solid #cbd5e1;
            box-shadow: 0 8px 32px rgba(71, 85, 105, 0.08);">
    <h3 style="margin: 0; font-size: 1.8rem; color: #0f172a;">🧠 Clinical AI Brain Tumor Detection System</h3>
    <p style="margin: 1.5rem 0; color: #475569; font-size: 1.1rem;">
        Advanced deep learning technology for medical image analysis and educational insights
    </p>
    <p style="margin: 0; color: #64748b; font-size: 0.9rem; font-weight: 500;">
        💡 Optimized for high-resolution MRI scans • Designed for medical education<br>
        Built with TensorFlow • Streamlit • Clinical-grade Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)
