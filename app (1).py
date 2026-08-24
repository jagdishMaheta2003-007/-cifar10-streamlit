import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# CIFAR-10 Image Classifier — Streamlit App
# ---------------------------------------------------------

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

MODEL_PATH = "cifar10_model.h5"  # exported from your Colab notebook


@st.cache_resource
def load_model():
    """Load the trained CIFAR-10 model once and cache it."""
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image into the (1, 32, 32, 3) array the model expects."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (32, 32))
    img_norm = img / 255.0
    return img_norm.reshape(1, 32, 32, 3), img


def main():
    st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🖼️", layout="centered")

    st.title("🖼️ CIFAR-10 Image Classifier")
    st.write(
        "Upload an image and the model will predict which of the 10 CIFAR-10 "
        "classes it belongs to: " + ", ".join(CLASS_NAMES)
    )

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Could not load '{MODEL_PATH}'. Make sure the trained model file "
            f"is in the same folder as app.py.\n\nDetails: {e}"
        )
        st.stop()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        input_array, display_img = preprocess_image(pil_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Predicting..."):
            preds = model.predict(input_array)
            pred_class = int(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

        with col2:
            st.image(display_img, caption="Resized to 32x32 (model input)", use_container_width=True)

        st.success(f"**Prediction: {CLASS_NAMES[pred_class].upper()}** ({confidence:.2f}% confidence)")

        st.subheader("Class probabilities")
        prob_dict = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(prob_dict)
    else:
        st.info("Upload a JPG or PNG image to get a prediction.")


if __name__ == "__main__":
    main()
