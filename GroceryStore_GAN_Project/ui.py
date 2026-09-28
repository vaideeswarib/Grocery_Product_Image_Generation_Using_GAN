import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Product Image Generator",
    page_icon="🛒",
    layout="wide"
)

st.title("Product Image Generation using Basic GAN")
st.write("Generate synthetic grocery product images using the trained Basic GAN Generator.")

api_url = st.text_input("FastAPI URL", "http://127.0.0.1:8000")

num_images = st.number_input(
    "Number of images",
    min_value=1,
    max_value=16,
    value=5,
    step=1
)

if st.button("Generate Images"):
    try:
        response = requests.post(
            f"{api_url}/generate",
            json={"num_images": int(num_images)},
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        st.success(f"Generated {result['num_images']} images.")

        columns = st.columns(4)
        for i, encoded in enumerate(result["images"]):
            image = Image.open(BytesIO(base64.b64decode(encoded)))
            with columns[i % 4]:
                st.image(image, caption=f"Generated Product {i + 1}", use_container_width=True)

    except requests.exceptions.RequestException as exc:
        st.error("Could not connect to FastAPI. Start the API first.")
        st.code(str(exc))
