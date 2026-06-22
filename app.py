
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import pandas as pd

from urdu_ocr import UrduOCR


st.set_page_config(
    page_title="Urdu Handwriting Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Urdu Handwriting Recognition")
st.markdown("Draw an Urdu character below — or upload an image — and the model will predict what it is.")
st.markdown("---")


# ============================================
# LOAD MODEL (only once, cached for speed)
# ============================================

@st.cache_resource
def load_ocr_model():
    """Loads and returns the UrduOCR model. Cached so it loads only once."""
    ocr = UrduOCR(model_path="urdu_model.h5")
    ocr.load()
    return ocr

ocr = load_ocr_model()

input_method = st.radio(
    "Choose input method:",
    ["✏️ Draw a character", "📁 Upload an image"],
    horizontal=True
)

img_array = None 

if input_method == "✏️ Draw a character":

    st.subheader("Draw your Urdu character here")
    st.caption("Use your mouse (or finger on touch screen) to draw. Click **Reset** to clear.")

    try:
        from streamlit_drawable_canvas import st_canvas

        if "canvas_key" not in st.session_state:
            st.session_state["canvas_key"] = 0

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Reset Canvas"):
                st.session_state["canvas_key"] += 1

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=12,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        # Convert canvas image to numpy array if user has drawn something
        if canvas_result.image_data is not None:
            canvas_img = canvas_result.image_data.astype("uint8")
            # Canvas gives RGBA; convert to grayscale
            if canvas_img.shape[2] == 4:
                canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)
            elif canvas_img.shape[2] == 3:
                canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2GRAY)

            # Only proceed if something was actually drawn (not a blank white canvas)
            if canvas_img.min() < 200:
                img_array = canvas_img

    except ImportError:
        st.warning(
            "The drawing canvas requires `streamlit-drawable-canvas`.\n\n"
            "Install it by running:\n\n"
            "```\npip install streamlit-drawable-canvas\n```\n\n"
            "Then restart the app. In the meantime, please use the **Upload an image** option."
        )

else:
    st.subheader("Upload a character image")
    st.info("For best results, upload images from:  `data/data/data/characters_test_set/<class>/`")

    uploaded_file = st.file_uploader(
        "Choose an image (JPG or PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded image", width=200)
        pil_image = Image.open(uploaded_file).convert("L")   # grayscale
        img_array = np.array(pil_image)


st.markdown("---")

if img_array is not None:

    if st.button("🔍 Predict Character", type="primary"):

        predicted_character, confidence, top5 = ocr.predict_from_array(img_array)

        if predicted_character is not None:

            st.subheader("Prediction Result")

            st.markdown(
                f"<h2 style='text-align:center; color:#1f77b4;'>"
                f"{predicted_character.upper()}</h2>",
                unsafe_allow_html=True
            )

            st.metric(label="Confidence", value=f"{confidence:.1f}%")
            st.progress(min(confidence / 100, 1.0))

            st.markdown("---")
            st.subheader("Top 5 Predictions")

            chart_data = pd.DataFrame({
                "Character": [item[0] for item in top5],
                "Confidence (%)": [item[1] for item in top5]
            }).set_index("Character")

            st.bar_chart(chart_data)

        else:
            st.error("Could not process the image. Please try again.")

else:
    if input_method == "✏️ Draw a character":
        st.info("Draw something on the canvas above, then click **Predict Character**.")
    else:
        st.info("Upload an image above, then click **Predict Character**.")


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "AIC270 — Programming for AI | SP24 BAI | COMSATS University"
    "</p>",
    unsafe_allow_html=True
)
