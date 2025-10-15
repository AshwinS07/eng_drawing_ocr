import sys
import os
import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from pipeline.run_pipeline import run

st.set_page_config(layout="wide")
st.title("📐 Engineering Drawing OCR (Labels + Values)")

uploaded_file = st.file_uploader("Upload an engineering drawing", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tfile.write(uploaded_file.read())
    tfile.close()

    outdir = ROOT_DIR / "outputs" / "streamlit"
    os.makedirs(outdir, exist_ok=True)

    st.info("Processing image… please wait ⏳")

    try:
        payload = run(str(tfile.name), out_dir=str(outdir))
    except Exception as e:
        st.error(f"Error running OCR pipeline: {e}")
        st.stop()

    # --- Show extracted labels & values ---
    st.subheader("🔍 Extracted Label – Value Pairs")
    if payload and "results" in payload and payload["results"]:
        df = pd.DataFrame(payload["results"])[["Label", "Value"]]
        st.dataframe(df, width="content")
    else:
        st.warning("⚠️ No labels or values detected.")

    # --- Show visualization image ---
    st.subheader("🖼️ Visualization with Bounding Boxes")
    viz_path = outdir / "viz.jpg"
    if viz_path.exists():
        st.image(str(viz_path), caption="Detected labels (blue) and values (green)")
    else:
        st.warning("Visualization image not found.")
