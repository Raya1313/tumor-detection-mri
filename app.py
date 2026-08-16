import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image

from gradcam import (
    get_models,
    prep_grad_model,
    PREPROCESS_FNS,
    CLASS_NAMES,
    get_gradcam_heatmap,
    resize_heatmap,
    overlay_heatmap,
    to_display_rgb,
)
from ensemble import predict_ensemble

IMG_SIZE = 224

st.set_page_config(page_title="MRI Tumor Classifier", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_everything():
    models = get_models()
    preprocess_fns = PREPROCESS_FNS
    grad_models = {name: prep_grad_model(obj, name) for name, obj in models.items()}
    meta_lr = joblib.load("models/meta_lr_ensemble.pkl")
    return models, preprocess_fns, grad_models, meta_lr


def load_image_as_array(uploaded_file):
    """Load an uploaded image file into a uint8 RGB array of shape (IMG_SIZE, IMG_SIZE, 3),
    matching the format the cached dataset (dataset_rgb.npz) stores images in."""
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.uint8)


st.title("MRI Tumor Classification")
st.caption("Ensemble of Base CNN + ResNet50 + EfficientNetB0, with Grad-CAM explainability")

models, preprocess_fns, grad_models, meta_lr = load_everything()

# Must match the exact column order used when meta_lr was trained
META_MODEL_ORDER = ['resnet50_mri_tumor', 'efficientNetB0_mri_tumor', 'BaseCNN_mri_tumor']

uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_array = load_image_as_array(uploaded_file)

    # Get per-model probabilities (strategy doesn't matter here, we just need all_model_probs)
    _, _, all_model_probs = predict_ensemble(
        image_array, models, preprocess_fns, strategy="average"
    )

    # Build meta-feature vector in the SAME order used during meta_lr training
    X_meta = np.concatenate(
        [all_model_probs[name].reshape(1, -1) for name in META_MODEL_ORDER], axis=1
    )

    final_pred_idx = meta_lr.predict(X_meta)[0]
    final_pred_probs = meta_lr.predict_proba(X_meta)[0]
    pred_class = CLASS_NAMES[final_pred_idx]
    confidence = final_pred_probs[final_pred_idx]

    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image_array, caption="Input scan", width=280)

    with col_result:
        st.subheader("Ensemble prediction (logistic regression)")
        st.metric(label="Predicted class", value=pred_class, delta=f"{confidence:.1%} confidence")

    st.divider()
    st.subheader("Per-model breakdown")

    breakdown_cols = st.columns(len(all_model_probs))
    for col, (name, probs) in zip(breakdown_cols, all_model_probs.items()):
        top_idx = np.argmax(probs)
        with col:
            st.caption(f"**{name}** — {CLASS_NAMES[top_idx]} ({probs[top_idx]:.1%})")
            chart_series = pd.Series(dict(zip(CLASS_NAMES, probs)))
            chart_series.index.name = "Class"
            chart_series.name = 'Value'
            st.bar_chart(chart_series, height=140)

    st.divider()
    st.subheader("Grad-CAM: where each model is looking")

    gradcam_cols = st.columns(len(models))
    for col, (name, model) in zip(gradcam_cols, models.items()):
        preprocess_fn = preprocess_fns[name]
        grad_model = grad_models[name]

        model_input = preprocess_fn(np.expand_dims(image_array, axis=0).copy())
        probs = all_model_probs[name]
        pred_idx = int(np.argmax(probs))

        heatmap = get_gradcam_heatmap(grad_model, model_input, class_index=pred_idx)
        heatmap_resized = resize_heatmap(heatmap, target_size=(IMG_SIZE, IMG_SIZE))

        raw_display = to_display_rgb(image_array)
        overlay = overlay_heatmap(heatmap_resized, raw_display, alpha=0.4)

        with col:
            st.markdown(f"**{name}**")
            st.image(overlay, caption=f"Pred: {CLASS_NAMES[pred_idx]}", width=220)
else:
    st.info("Upload an MRI image to get a prediction.")