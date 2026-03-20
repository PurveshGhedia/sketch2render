# Sketch2Render

A real-time microservice that converts hand-drawn sketches into photorealistic images using a fine-tuned generative model.

## Stack

- **Model:** img2img-turbo (fine-tuned)
- **Backend:** FastAPI + PyTorch (MPS accelerated)
- **Frontend:** HTML5 Canvas
- **Infra:** Docker + Nginx

## Setup

```bash
conda env create -f environment.yml
conda activate sketch2render
```
