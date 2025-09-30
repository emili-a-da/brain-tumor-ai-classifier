# 🧠 Brain Tumor AI Classifier

A state-of-the-art AI-powered brain tumor classification system built with deep learning and Streamlit, designed for medical professionals and educational purposes.

## ✨ Features

- **🎯 Advanced AI Classification**: Detects 4 types of brain conditions
  - Glioma
  - Meningioma  
  - Normal Tissue
  - Pituitary Adenoma

- **⚕️ Professional Medical Interface**: Clinical-grade UI with pastel colors suitable for healthcare professionals
- **🎨 Animated UI**: Beautiful transitions and animations for enhanced user experience
- **📊 Detailed Analysis**: Comprehensive medical explanations for each tumor type
- **🔒 Privacy Compliant**: HIPAA-compliant design principles
- **⚡ Real-time Processing**: Instant analysis of MRI scans

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras with VGG16 architecture
- **Frontend**: Streamlit with custom CSS animations
- **Image Processing**: PIL, OpenCV
- **Deployment**: Streamlit Cloud ready

## 📋 Requirements

```
streamlit>=1.25.0
tensorflow>=2.12.0
Pillow>=9.5.0
numpy>=1.24.0
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-ai-classifier.git
   cd brain-tumor-ai-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run secrets.toml/streamlit_app_new.py
   ```

4. **Open your browser**
   - Navigate to `http://localhost:8501`

## 📁 Project Structure

```
brain-tumor-ai-classifier/
├── secrets.toml/
│   ├── streamlit_app_new.py      # Main Streamlit application
│   └── Zuzik_mri_model_final22.h5 # Trained AI model
├── utils.py                      # Utility functions
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🎯 Model Performance

- **Architecture**: Deep Convolutional Neural Network (VGG16-based)
- **Input Size**: 224×224 RGB images
- **Classes**: 4 (Glioma, Meningioma, Normal, Pituitary)
- **Training**: Medical-grade dataset with data augmentation

## 📖 Usage Instructions

1. **Upload MRI Image**: Click the upload area and select a brain MRI scan
2. **Wait for Analysis**: The AI model will process the image in real-time
3. **View Results**: Get instant classification with confidence scores
4. **Read Medical Information**: Review detailed explanations in organized tabs
5. **Analyze Confidence**: Check detailed probability breakdown

## ⚠️ Medical Disclaimer

**IMPORTANT**: This AI model is designed for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for proper medical evaluation, diagnosis, and treatment planning.


## 🙏 Acknowledgments

- Medical imaging datasets used for training
- TensorFlow/Keras community
- Streamlit framework developers
- Healthcare professionals who provided domain expertise



---


⚕️ **Built for Healthcare Innovation** • 🧠 **Powered by AI** • 🎯 **Designed for Accuracy**


