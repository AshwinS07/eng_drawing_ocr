import imghdr  # ✅ Fix for Python 3.12+ Streamlit image issue


import logging
import os

# Suppress PyTorch warnings on Streamlit Cloud
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
logging.getLogger("torch").setLevel(logging.ERROR)

import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import sys
import os
import json
from io import BytesIO
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from src.recognition.text_recognizer import TextRecognizer

# Page config
st.set_page_config(
    page_title="Engineering Drawing OCR",
    page_icon="📐",
    layout="wide"
)

# Initialize session state
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'image' not in st.session_state:
    st.session_state.image = None


def initialize_recognizer():
    """Initialize the text recognizer"""
    if st.session_state.recognizer is None:
        with st.spinner("Initializing OCR engine..."):
            st.session_state.recognizer = TextRecognizer(lang='en')
    return st.session_state.recognizer


def process_image(image):
    """Process uploaded image"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_cv


def main():
    st.title("📐 Engineering Drawing OCR System")
    st.markdown("### Extract dimensions and labels from technical drawings")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Detection parameters
        st.subheader("Detection Parameters")
        max_distance = st.slider(
            "Max pairing distance",
            50, 500, 200,
            help="Maximum distance for label-value pairing"
        )

        confidence_threshold = st.slider(
            "Confidence threshold",
            0.0, 1.0, 0.5,
            help="Minimum confidence for text detection"
        )

        # Visualization options
        st.subheader("Visualization")
        show_bbox = st.checkbox("Show bounding boxes", value=True)
        show_connections = st.checkbox("Show label-value connections", value=True)
        show_confidence = st.checkbox("Show confidence scores", value=True)

        # Export options
        st.subheader("Export")
        export_format = st.selectbox(
            "Export format",
            ["JSON", "CSV", "Excel", "All"]
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Drawing")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a technical drawing or engineering diagram"
        )

        if uploaded_file is not None:
            try:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
            except UnidentifiedImageError:
                st.error("❌ Unable to open the uploaded file. Please upload a valid image (jpg, png, bmp, tiff).")
                return

            # Process button
            if st.button("🔍 Extract Dimensions", type="primary"):
                with st.spinner("Processing image..."):
                    # Initialize recognizer
                    recognizer = initialize_recognizer()

                    # Convert and process image
                    img_cv = process_image(image)
                    st.session_state.image = img_cv

                    # Extract dimensions
                    results = recognizer.extract_dimensions(img_cv)
                    st.session_state.results = results

                    st.success(f"✅ Detected {len(results['text_data'])} text elements!")

    with col2:
        st.header("Results")

        if st.session_state.results is not None:
            results = st.session_state.results

            # Statistics
            st.subheader("📊 Statistics")
            stat_cols = st.columns(4)
            with stat_cols[0]:
                st.metric("Total Text", len(results['text_data']))
            with stat_cols[1]:
                st.metric("Labels", len(results['labels']))
            with stat_cols[2]:
                st.metric("Values", len(results['values']))
            with stat_cols[3]:
                st.metric("Pairs", len(results['pairs']))

            # Visualization
            st.subheader("🎨 Visualization")
            vis_image = st.session_state.recognizer.visualize_results(
                st.session_state.image.copy(),
                results
            )
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            st.image(vis_image_rgb, caption="Detected Text", use_container_width=True)

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Pairs", "📝 All Text", "📈 Details", "💾 Export"
            ])

            with tab1:
                st.subheader("Label-Value Pairs")

                # Filter by confidence
                filtered_pairs = [
                    p for p in results['pairs']
                    if p['confidence'] >= confidence_threshold
                ]

                if filtered_pairs:
                    pairs_df = pd.DataFrame([
                        {
                            'Label': p['label'],
                            'Value': p['value'] if p['value'] else 'N/A',
                            'Confidence': f"{p['confidence']:.2%}"
                        }
                        for p in filtered_pairs
                    ])
                    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

                    # Download as CSV
                    csv = pairs_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Pairs as CSV",
                        csv,
                        "pairs.csv",
                        "text/csv"
                    )
                else:
                    st.info("No pairs found above confidence threshold")

            with tab2:
                st.subheader("All Detected Text")

                filtered_text = [
                    item for item in results['text_data']
                    if item['confidence'] >= confidence_threshold
                ]

                if filtered_text:
                    text_df = pd.DataFrame([
                        {
                            'Text': item['text'],
                            'Type': item['type'],
                            'Confidence': f"{item['confidence']:.2%}",
                            'Position': f"({int(item['center'][0])}, {int(item['center'][1])})"
                        }
                        for item in filtered_text
                    ])
                    st.dataframe(text_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No text found above confidence threshold")

            with tab3:
                st.subheader("Detection Details")

                for idx, item in enumerate(results['text_data'][:20]):
                    with st.expander(f"{idx+1}. {item['text']} ({item['type']})"):
                        detail_cols = st.columns(2)
                        with detail_cols[0]:
                            st.write(f"**Text:** {item['text']}")
                            st.write(f"**Type:** {item['type']}")
                            st.write(f"**Confidence:** {item['confidence']:.2%}")
                        with detail_cols[1]:
                            st.write(f"**Center:** ({int(item['center'][0])}, {int(item['center'][1])})")
                            st.write(f"**Width:** {int(item['width'])}")
                            st.write(f"**Height:** {int(item['height'])}")

            with tab4:
                st.subheader("Export Results")

                export_data = {
                    'statistics': {
                        'total_text': len(results['text_data']),
                        'labels': len(results['labels']),
                        'values': len(results['values']),
                        'pairs': len(results['pairs'])
                    },
                    'pairs': [
                        {
                            'label': p['label'],
                            'value': p['value'],
                            'confidence': float(p['confidence'])
                        }
                        for p in results['pairs']
                    ],
                    'all_text': [
                        {
                            'text': item['text'],
                            'type': item['type'],
                            'confidence': float(item['confidence'])
                        }
                        for item in results['text_data']
                    ]
                }

                # JSON export
                if export_format in ["JSON", "All"]:
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_str,
                        "results.json",
                        "application/json"
                    )

                # CSV export
                if export_format in ["CSV", "All"]:
                    pairs_df = pd.DataFrame(export_data['pairs'])
                    csv = pairs_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "results.csv",
                        "text/csv"
                    )

                # Excel export
                if export_format in ["Excel", "All"]:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        pd.DataFrame(export_data['pairs']).to_excel(
                            writer, sheet_name='Pairs', index=False
                        )
                        pd.DataFrame(export_data['all_text']).to_excel(
                            writer, sheet_name='All Text', index=False
                        )

                    st.download_button(
                        "📥 Download Excel",
                        output.getvalue(),
                        "results.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # Download visualization
                st.subheader("Download Visualization")
                vis_buffer = BytesIO()
                vis_image_pil = Image.fromarray(vis_image_rgb)
                vis_image_pil.save(vis_buffer, format='PNG')
                st.download_button(
                    "📥 Download Annotated Image",
                    vis_buffer.getvalue(),
                    "annotated_drawing.png",
                    "image/png"
                )

        else:
            st.info("👆 Upload an image and click 'Extract Dimensions' to see results")

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** For best results, use high-resolution images with clear text. "
        "Adjust the confidence threshold in the sidebar to filter detections."
    )


if __name__ == "__main__":
    main()
