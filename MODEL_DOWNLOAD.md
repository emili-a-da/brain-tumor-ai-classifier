# Model File Location

The trained model file `Zuzik_mri_model_final22.h5` is too large for direct GitHub upload (>100MB).

## Download the Model

To run this application locally, you'll need to download the model file:

1. **Download Link**: [Add your Google Drive/Dropbox link here]
2. **File Size**: ~50-100MB
3. **Placement**: Save as `secrets.toml/Zuzik_mri_model_final22.h5`

## Alternative: Use Streamlit Secrets

You can also configure the app to download the model automatically by adding this to your Streamlit secrets:

```toml
MODEL_URL = "https://your-download-link-here.com/Zuzik_mri_model_final22.h5"
```

The app will automatically download and cache the model on first run.