# Product Image Generation using Basic GAN

## 1. Project Overview

**Problem:** Retail and grocery applications may require large collections of product images for prototyping, computer-vision experimentation, testing and synthetic-data research.

**Objective:** Train a Basic Generative Adversarial Network using the GroceryStoreDataset and generate synthetic grocery product images.

**Real-world application:** Synthetic product images can support catalog prototyping, image-pipeline testing and data-augmentation research. Generated images are synthetic and are not verified product photographs.

## 2. Dataset

Dataset: GroceryStoreDataset  
Source: https://github.com/marcusklasson/GroceryStoreDataset

The approved notebook performs image validation, EDA, leakage-aware splitting and preprocessing.

GAN input: random latent vector of dimension 100.  
GAN output: 64×64 RGB synthetic product image.

## 3. Technology Stack

Python, NumPy, Pandas, Matplotlib, Pillow, TensorFlow/Keras, FastAPI, Uvicorn, Streamlit, Docker, Git/GitHub.

## 4. Workflow

```text
Dataset
→ Dataset Understanding
→ EDA
→ Data Quality
→ Train/Validation/Test Split
→ 64×64 RGB Preprocessing
→ Basic GAN
→ Training
→ Experiments
→ Evaluation
→ Final Generator
→ FastAPI
→ JSON Output
→ Docker
```

## 5. Model Architecture

### Generator

```text
Latent 100
→ Dense 8×8×256
→ BatchNorm
→ ReLU
→ Reshape
→ Conv2DTranspose 128
→ BatchNorm
→ ReLU
→ Conv2DTranspose 64
→ BatchNorm
→ ReLU
→ Conv2DTranspose 3
→ Tanh
→ 64×64×3
```

### Discriminator

```text
64×64×3
→ Conv2D 64
→ LeakyReLU(0.2)
→ Dropout(0.30)
→ Conv2D 128
→ LeakyReLU(0.2)
→ Dropout(0.30)
→ Conv2D 256
→ LeakyReLU(0.2)
→ Dropout(0.30)
→ Flatten
→ Dense(1)
→ Sigmoid
```

### Approved configuration

| Parameter | Value |
|---|---:|
| Image size | 64×64 |
| Channels | RGB |
| Latent dimension | 100 |
| Batch size | 16 |
| Epochs | 30 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Beta1 | 0.5 |
| Loss | Binary Cross-Entropy |
| Generator output | Tanh |

## 6. Experiments

1. Baseline: LR 0.0002, Beta1 0.5, Dropout 0.30, Latent 100
2. Dropout 0.40: LR 0.0002, Beta1 0.5, Dropout 0.40, Latent 100
3. LR 0.0001 + Latent 128: LR 0.0001, Beta1 0.5, Dropout 0.30, Latent 128

## 7. Evaluation

Generator loss, discriminator loss, FID, KID, brightness, contrast, diversity, generated-image visualization, stability and mode-collapse observations.

GAN losses are interpreted together with visual quality and diversity rather than as an accuracy score.

## 8. Deployment

FastAPI endpoints:

- `GET /`
- `GET /health`
- `POST /generate`

Request:

```json
{"num_images": 5}
```

Response contains JSON metadata and Base64-encoded PNG images.

Swagger UI:

`http://localhost:8000/docs`

## 9. UI

`ui.py` provides a lightweight Streamlit interface that calls FastAPI.

```bash
streamlit run ui.py
```

## 10. Docker

Build:

```bash
docker build -t product-gan-api .
```

Run:

```bash
docker run --rm -p 8000:8000 product-gan-api
```

## 11. Limitations

- 64×64 resolution
- possible GAN artifacts
- possible mode collapse
- training instability
- CPU training/generation is slower than GPU execution
- synthetic images should not be treated as verified product photographs

## 12. Future Scope

Higher resolution, larger datasets, conditional generation, improved GAN architectures, stronger diversity evaluation, GPU training and production monitoring.

## 13. Project Structure

```text
Project_Name/
├── Project_Notebook.ipynb
├── generator.keras
├── discriminator.keras
├── app.py
├── ui.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── model/
│   ├── generator.keras
│   └── discriminator.keras
├── generated_images/
│   ├── epoch_1.png
│   ├── epoch_10.png
│   ├── epoch_20.png
│   ├── epoch_30.png
│   └── final_grid.png
├── Screenshots/
└── outputs/
```

## 14. Final demonstration

Problem → Dataset → EDA → Preprocessing → Representation → Generator → Discriminator → Training → Experiments → Evaluation → Error Analysis → FastAPI → JSON generation → Docker → UI → Limitations → Future scope.
