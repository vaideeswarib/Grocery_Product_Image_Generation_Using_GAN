from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

IMAGE_SIZE = 64
LATENT_DIM = 100
MAX_IMAGES = 16
GENERATION_BATCH_SIZE = 4

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "generator.keras"

app = FastAPI(
    title="Product Image Generation - Basic GAN API",
    description="Generate synthetic grocery product images using the trained Basic GAN Generator.",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained Generator not found: {MODEL_PATH}. "
        "Place generator.keras inside the model folder."
    )

generator = tf.keras.models.load_model(MODEL_PATH, compile=False)


class GenerateRequest(BaseModel):
    num_images: int = Field(default=1, ge=1, le=MAX_IMAGES)


def tensor_to_base64_png(image_tensor):
    image = (image_tensor + 1.0) / 2.0
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)

    pil_image = Image.fromarray(image, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
def root():
    return {
        "project": "Product Image Generation using Basic GAN",
        "status": "running",
        "endpoint": "POST /generate"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "image_size": IMAGE_SIZE,
        "latent_dimension": LATENT_DIM,
        "max_images_per_request": MAX_IMAGES
    }


@app.post("/generate")
def generate_images(request: GenerateRequest):
    try:
        num_images = int(request.num_images)
    except Exception:
        raise HTTPException(status_code=400, detail="num_images must be an integer.")

    if not 1 <= num_images <= MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"num_images must be between 1 and {MAX_IMAGES}."
        )

    images = []

    # Small batches reduce peak RAM usage.
    for start in range(0, num_images, GENERATION_BATCH_SIZE):
        current_size = min(GENERATION_BATCH_SIZE, num_images - start)

        latent_vectors = tf.random.normal(
            shape=(current_size, LATENT_DIM)
        )

        generated = generator(latent_vectors, training=False).numpy()

        for image_tensor in generated:
            images.append(tensor_to_base64_png(image_tensor))

        del latent_vectors
        del generated

    return {
        "status": "success",
        "num_images": len(images),
        "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "format": "PNG",
        "encoding": "base64",
        "images": images
    }
