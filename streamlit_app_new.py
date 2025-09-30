# streamlit_app.py - Main Application Entry Point
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

# Model path - now looks in the secrets.toml folder
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "secrets.toml", "Zuzik_mri_model_final22.h5")

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
    MODEL_URL = ""

# For now, redirect users to run the app from the secrets.toml folder
st.error("⚠️ **Important**: Please run the app using the file in the secrets.toml folder")
st.code("streamlit run secrets.toml/streamlit_app_new.py")
st.info("This is the main entry point, but the full app is located in the secrets.toml folder.")
