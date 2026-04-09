from pix2pix_turbo import Pix2Pix_Turbo
import sys
import os
import torch
import io
import base64
import numpy as np
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import ImageOps

# Point Python to the cloned directory
sys.path.append(os.path.join(os.path.dirname(
    __file__), '..', 'img2img-turbo', 'src'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def load_model():
    global model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Initializing model on {device}...")

    model = Pix2Pix_Turbo(
        pretrained_name="stabilityai/sd-turbo",
        pretrained_path="checkpoints/New/model_5001.pkl"
    )

    # THE MEMORY BINDING FIX: Split the cast and the move!
    # 2. Cast to float32 safely on the CPU
    model.to(torch.float32)
    # 3. NOW move the prepared weights to the Mac GPU
    model.to(device)

    model.eval()
    print(f"✅ Model successfully loaded on {device} in float32!")


class GenerateRequest(BaseModel):
    image: str


class GenerateResponse(BaseModel):
    image: str


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    device = next(model.parameters()).device

    # ==========================================
    # INPUT HANDLING
    # ==========================================
    # OPTION A: Test with local canny file (Commented out for live usage)
    # img = Image.open("data/test_pairs/canny.jpg").convert("RGB").resize((512, 512))

    # OPTION B: Normal live UI path (Canvas Input)
    img_str = request.image
    # Strip the data URI prefix if the frontend sends it (e.g., "data:image/png;base64,")
    if "base64," in img_str:
        img_str = img_str.split("base64,")[1]

    # Decode the base64 string back into bytes
    img_bytes = base64.b64decode(img_str)
    # Open as PIL Image, ensure RGB format, and lock to 512x512
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((512, 512))
    # ==========================================

    # --- THE FIX: Pure RGB Preprocessing ---
    # --- THE FIX 2.0: [0, 1] Sketch Normalization ---
    # Convert directly to numpy, keep RGB, permute to PyTorch format (C,H,W)
    img_tensor = torch.from_numpy(np.array(img)).permute(
        2, 0, 1).unsqueeze(0).float()

    # Normalize strictly to [0.0, 1.0] for the conditioning sketch!
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.to(device, dtype=torch.float32)

    with torch.no_grad():
        # Pix2Pix-turbo inference call
        output = model(
            img_tensor, prompt="a photorealistic portrait of a person")

    print(
        f"Tensor Health -> Min: {output.min().item():.2f}, Max: {output.max().item():.2f}")

    # --- Postprocessing ---
    # Squeeze batch dim, permute back to (H,W,C)
    output_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    # Denormalize from [-1, 1] back to [0, 1]
    output_np = (output_np + 1.0) / 2.0
    # Scale to [0, 255] and cast to uint8
    output_img = (output_np * 255).clip(0, 255).astype(np.uint8)

    # Output pure RGB image
    pil_img = Image.fromarray(output_img)
    pil_img.save("debug_final_output.png")  # Save a test copy locally

    # Encode for API response
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return GenerateResponse(image=encoded)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
