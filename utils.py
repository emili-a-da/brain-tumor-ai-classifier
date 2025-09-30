import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess

# utils.py
import json, os
def load_labels(model_dir):
    with open(os.path.join(model_dir, "labels.json")) as f:
        return json.load(f)


# Must match the label order used when creating image_dataset_from_directory
LABELS = ['glioma','meningioma','notumor','pituitary']  
IMG_SIZE = (224, 224)

def load_model(path="e:\\MRI_Project_5\\mri_project_5\\Zuzik_mri_model_final22.h5"):
    """
    Load a Keras model. compile=False is faster and avoids optimizer/loss deps.
    If you trained with custom layers, pass custom_objects accordingly.
    """
    model = tf.keras.models.load_model(path, compile=False,safe_mode=False)
    return model

def _open_image(src):
    """
    Accepts either a path (str/PathLike) or a file-like object (e.g., Flask's FileStorage).
    Returns a PIL.Image in RGB mode.
    """
    if hasattr(src, "read"):  # file-like
        try:
            # If it's a stream, rewind just in case it was read earlier
            if hasattr(src, "seek"):
                src.seek(0)
            img = Image.open(src)
        except Exception:
            # Some frameworks give bytes; wrap in BytesIO and retry
            data = src.read()
            img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(str(src))
    return img.convert("RGB")

def preprocess_image(src):
    """
    Convert input image to a (1, 224, 224, 3) float32 tensor preprocessed for VGG16.
    This MUST mirror your training pipeline (VGG16 preprocess_input on RGB).
    """
    img = _open_image(src).resize(IMG_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)           # (224, 224, 3), RGB
    arr = vgg16_preprocess(arr)                       # mean-subtract, BGR ordering handled internally
    arr = np.expand_dims(arr, axis=0)                 # (1, 224, 224, 3)
    return arr

def postprocess_prediction(pred):
    """
    pred: model output from model.predict(x), expected shape (1, num_classes)
    Returns: (top_label: str, top_conf: float, probs: np.ndarray)
    - Does NOT apply softmax again if the model already has softmax (yours does).
    """
    pred = np.asarray(pred)
    if pred.ndim != 2 or pred.shape[0] != 1:
        # Try to coerce to (1, C)
        pred = pred.reshape(1, -1)

    probs = pred[0]

    # If the last layer wasn't softmax (not your case), normalize safely:
    if not np.allclose(probs.sum(), 1.0, atol=1e-3) or np.any(probs < 0):
        # apply softmax only if it doesn't look like probabilities
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    if len(LABELS) != probs.shape[-1]:
        raise ValueError(
            f"Label length ({len(LABELS)}) != model outputs ({probs.shape[-1]}). "
            "Ensure LABELS order matches the training class order."
        )

    idx = int(np.argmax(probs))
    label = LABELS[idx]
    confidence = float(probs[idx])
    return label, confidence, probs
