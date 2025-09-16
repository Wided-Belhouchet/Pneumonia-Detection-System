# ----------------------------------
# Import Libraries
# -----------------------------------
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import matplotlib.pyplot as plt

# -----------------------------------
# Set Background
# -----------------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        .stButton>button {{
            background-color: #2e86de;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 16px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg5.png")

# -----------------------------------
# Title & Instructions
# -----------------------------------
st.markdown("<h1 style='text-align: center; color: #1e3799;'>🩺 Pneumonia Detection System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Upload a chest X-ray image and let our AI model classify it as <b>Normal</b> or <b>Pneumonia</b>.</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------
# Upload file
# -----------------------------------
file = st.file_uploader(
    " Upload a Chest X-ray Image",
    type=['jpeg', 'jpg', 'png'],
    label_visibility="visible"
)

# -----------------------------------
# Load Trained Model
# -----------------------------------
model = load_model("resnet50_pneumonia_model.h5")

# Class names
class_names = ["Normal", "Pneumonia"]

# -----------------------------------
# Prediction
# -----------------------------------
if file is not None:
    image = Image.open(file).convert('RGB')

    # Left/Right layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, use_column_width=True, caption="Uploaded X-ray")

    with col2:
        st.markdown("### Model Prediction")

        # Preprocess image
        img_resized = image.resize((256, 256))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]

        if prediction < 0.5:
            predicted_class = class_names[0]  # Normal
            confidence = (1 - prediction) * 100
            color = "green"
        else:
            predicted_class = class_names[1]  # Pneumonia
            confidence = prediction * 100
            color = "red"

        # Show result
        st.markdown(f"<h2 style='color:{color};'>{predicted_class}</h2>", unsafe_allow_html=True)
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Confidence Bar
        st.progress(int(confidence))

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([confidence, 100 - confidence], labels=[predicted_class, "Other"],
               autopct='%1.1f%%', colors=[color, "#dcdde1"], startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

# -----------------------------------
# Footer
# -----------------------------------
st.divider()
st.markdown("<p style='text-align: center; font-size:14px;'>Developed with ❤️ using Streamlit & TensorFlow</p>", unsafe_allow_html=True)
