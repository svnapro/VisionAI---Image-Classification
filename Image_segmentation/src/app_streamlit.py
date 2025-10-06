import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
import io
import torch

from src.model import get_maskrcnn, get_resnet50
from src.utils import pil_to_tensor, refine_with_classifier, make_zip_bytes

# --- Setup ---
st.set_page_config(page_title="Object Detection + Classifier", layout="centered")
st.title("🐶 Hybrid Object Detection (Mask R-CNN + ResNet50)")
st.write("Upload images. Detector finds objects, classifier refines labels (better for dogs/cats/horses).")

# Sidebar controls
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Running on: **{device}**")
confidence_threshold = st.sidebar.slider("Mask R-CNN confidence threshold", 0.0, 1.0, 0.8, 0.05)

# Cache models
@st.cache_resource
def load_models():
    detector = get_maskrcnn(device=device)
    classifier = get_resnet50(device=device)
    return detector, classifier

detector, classifier = load_models()

# File uploader
uploaded_files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    download_bufs, download_names = [], []

    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📷 {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        # Detection
        x = pil_to_tensor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = detector(x)[0]

        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        st.write("### Detected Objects:")
        summary = []
        for box, sc in zip(boxes, scores):
            if sc < confidence_threshold: 
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = image.crop((x1, y1, x2, y2))

            # Refine label with classifier
            refined_label, refined_conf = refine_with_classifier(crop, classifier, device)

            st.write(f"👉 This is a **{refined_label}** (confidence {refined_conf:.2f})")
            summary.append(refined_label)

        if summary:
            st.info("Summary: " + ", ".join(summary))
        else:
            st.warning("No objects detected above threshold.")

        # Save for download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        fname = f"annotated_{uploaded_file.name.rsplit('.',1)[0]}.png"
        st.download_button("⬇️ Download image", data=buf, file_name=fname, mime="image/png")

        download_bufs.append(buf)
        download_names.append(fname)

    if len(download_bufs) > 1:
        zip_bytes = make_zip_bytes(download_bufs, download_names)
        st.download_button("⬇️ Download All (ZIP)", data=zip_bytes,
                           file_name="all_results.zip", mime="application/zip")
