from pix2pix_turbo import Pix2Pix_Turbo
import io
from PIL import Image
import numpy as np
import torch
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import sys
import os
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

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
    model = Pix2Pix_Turbo(pretrained_name="stabilityai/sd-turbo")

    checkpoint = torch.load("checkpoints/model_5501.pkl", map_location="cpu")

    # get LoRA config from checkpoint
    rank_unet = checkpoint["rank_unet"]
    rank_vae = checkpoint["rank_vae"]
    unet_lora_modules = checkpoint["unet_lora_target_modules"]
    vae_lora_modules = checkpoint["vae_lora_target_modules"]

    # apply LoRA to unet
    unet_lora_config = LoraConfig(
        r=rank_unet, target_modules=unet_lora_modules)
    model.unet = get_peft_model(model.unet, unet_lora_config)
    set_peft_model_state_dict(model.unet, checkpoint["state_dict_unet"])

    # apply LoRA to vae
    vae_lora_config = LoraConfig(r=rank_vae, target_modules=vae_lora_modules)
    model.vae = get_peft_model(model.vae, vae_lora_config)
    set_peft_model_state_dict(model.vae, checkpoint["state_dict_vae"])

    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")


class GenerateRequest(BaseModel):
    image: str  # base64 encoded PNG


class GenerateResponse(BaseModel):
    image: str  # base64 encoded PNG


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256))

    device = next(model.parameters()).device
    img_tensor = torch.from_numpy(np.array(img)).permute(
        2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    img_tensor = img_tensor.to(device)

    prompt = "a photorealistic portrait of a person"
    tokens = model.tokenizer(
        prompt,
        max_length=model.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)

    with torch.no_grad():
        output = model(img_tensor, prompt_tokens=tokens, deterministic=True)

    print("Output min:", output.min().item())
    print("Output max:", output.max().item())
    print("Output mean:", output.mean().item())
    print("Output shape:", output.shape)

    output_img = ((output.squeeze().permute(1, 2, 0).cpu(
    ).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(output_img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return GenerateResponse(image=encoded)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
