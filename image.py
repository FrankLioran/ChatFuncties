# image.py — Hugging Face compatibele versie
import requests
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import os
import streamlit as st
import base64
import random

# ---------------------------------------------------------
# 1. Cross‑platform Pollinations input directory
# ---------------------------------------------------------
BASE_DIR = Path(__file__).parent
POLLINATIONS_INPUT_DIR = BASE_DIR / "pollinations_input"
POLLINATIONS_INPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 2. ComfyUI uitgeschakeld (werkt niet op Hugging Face)
# ---------------------------------------------------------
def start_comfyui():
    st.warning("ComfyUI kan niet worden gestart op Hugging Face Spaces.")
    return None

def generate_comfy_image(*args, **kwargs):
    st.warning("Lokale ComfyUI beeldgeneratie is niet beschikbaar op Hugging Face.")
    return None

def comfyui_generate_video_from_image(*args, **kwargs):
    st.warning("Video-generatie via ComfyUI is niet beschikbaar op Hugging Face.")
    return None

# ---------------------------------------------------------
# 3. Pollinations — werkt wél op Hugging Face
# ---------------------------------------------------------
def generate_pollinations(prompt: str) -> bytes:
    url = f"https://image.pollinations.ai/prompt/{prompt}"
    return requests.get(url).content

def save_pollinations_image(prompt):
    url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
    img_bytes = requests.get(url).content

    filename = prompt.replace(" ", "_") + ".png"
    save_path = POLLINATIONS_INPUT_DIR / filename

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    return save_path

def build_pollinations_url(
    prompt: str,
    model: str = None,
    ar: str = None,
    seed: int = None,
    negative: str = None,
    steps: int = None,
    image_url: str = None
):
    base = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
    params = []

    if model: params.append(f"model={model}")
    if ar: params.append(f"ar={ar}")
    if seed: params.append(f"seed={seed}")
    if negative: params.append(f"np={negative.replace(' ', '%20')}")
    if steps: params.append(f"steps={steps}")
    if image_url: params.append(f"image={image_url}")

    return base + ("?" + "&".join(params) if params else "")

def fetch_pollinations_image(url: str) -> bytes:
    try:
        response = requests.get(url)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", "").lower():
            return None
        return response.content
    except:
        return None

def save_to_comfyui_input(img_bytes: bytes, filename: str) -> str:
    save_path = POLLINATIONS_INPUT_DIR / filename
    try:
        with open(save_path, "wb") as f:
            f.write(img_bytes)
        return save_path
    except:
        return None

def generate_pollinations_image(
    prompt: str,
    model: str = None,
    ar: str = None,
    seed: int = None,
    negative: str = None,
    steps: int = None,
    image_url: str = None
):
    url = build_pollinations_url(prompt, model, ar, seed, negative, steps, image_url)
    img_bytes = fetch_pollinations_image(url)
    if img_bytes is None:
        return None, None

    filename = f"{prompt.replace(' ', '_')}.png"
    save_path = save_to_comfyui_input(img_bytes, filename)

    return save_path, img_bytes

def upload_image_for_pollinations(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ---------------------------------------------------------
# 4. Stijl-injecties (blijven werken)
# ---------------------------------------------------------
def generate_pollinations_styled(positive_prompt: str, negative_prompt: str = "", style_injection: str = ""):
    full_prompt = positive_prompt
    if style_injection: full_prompt += f", {style_injection}"
    if negative_prompt: full_prompt += f", avoid: {negative_prompt}"

    url = f"https://image.pollinations.ai/prompt/{full_prompt.replace(' ', '%20')}"
    img_bytes = fetch_pollinations_image(url)
    if not img_bytes:
        return None, None

    filename = full_prompt.replace(" ", "_") + ".png"
    save_path = save_to_comfyui_input(img_bytes, filename)

    return save_path, img_bytes

# Stijl presets
def poll_style_anime(p, n=""): return generate_pollinations_styled(p, n, "anime style, cel shading")
def poll_style_cinematic(p, n=""): return generate_pollinations_styled(p, n, "cinematic lighting, 35mm film")
def poll_style_realistic(p, n=""): return generate_pollinations_styled(p, n, "ultra realistic, photorealistic")
def poll_style_flux(p, n=""): return generate_pollinations_styled(p, n, "flux style, soft gradients")
def poll_style_sdxl(p, n=""): return generate_pollinations_styled(p, n, "sdxl style, detailed textures")

# Aspect ratio presets
def poll_ar_square(p, n=""): return generate_pollinations_styled(p + ", 1:1 aspect ratio", n)
def poll_ar_wide(p, n=""): return generate_pollinations_styled(p + ", 16:9 aspect ratio", n)
def poll_ar_vertical(p, n=""): return generate_pollinations_styled(p + ", 9:16 aspect ratio", n)

# Negative presets
def poll_negative_no_text(p): return generate_pollinations_styled(p, "text, watermark, logo")
def poll_negative_no_blur(p): return generate_pollinations_styled(p, "blurry, soft focus")

# Seed presets
def poll_seed_random(p, n=""): return generate_pollinations_styled(p + f", seed {random.randint(1,999999)}", n)
def poll_seed_fixed(p, n=""): return generate_pollinations_styled(p + ", seed 12345", n)

# Steps presets
def poll_steps_fast(p, n=""): return generate_pollinations_styled(p + ", fast generation", n)
def poll_steps_quality(p, n=""): return generate_pollinations_styled(p + ", high detail", n)

# Image-to-image
def poll_image_to_image(p, n, image_bytes):
    url = upload_image_for_pollinations(image_bytes)
    return generate_pollinations_styled(p + ", based on reference image", n, f"reference image: {url}")

# ---------------------------------------------------------
# 5. Pillow beeldbewerking — werkt volledig
# ---------------------------------------------------------
def apply_brightness(image, value): return ImageEnhance.Brightness(image).enhance(value)
def apply_contrast(image, value): return ImageEnhance.Contrast(image).enhance(value)
def apply_saturation(image, value): return ImageEnhance.Color(image).enhance(value)

def apply_sharpness(image, value):
    if value > 1.0:
        return image.filter(ImageFilter.SHARPEN)
    elif value < 1.0:
        return image.filter(ImageFilter.BLUR)
    return image

def apply_rotate(image, degrees): return image.rotate(degrees, expand=True)
def apply_flip(image, direction): return image.transpose(Image.FLIP_LEFT_RIGHT if direction=="Horizontaal" else Image.FLIP_TOP_BOTTOM)

def apply_sepia(image, intensity=0.7):
    img_rgb = image.convert("RGB")
    pixels = np.array(img_rgb)
    sepia_kernel = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_pixels = pixels.dot(sepia_kernel.T)
    sepia_pixels = np.clip(sepia_pixels, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_pixels)

def apply_effect(image, effect_name):
    if effect_name == "Zwart-Wit":
        return image.convert('L').convert('RGBA')
    if effect_name == "Sepia":
        return apply_sepia(image)
    if effect_name == "Gaussiaanse Vervaag (Blur)":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if effect_name == "Scherpe Schets (Detail)":
        return image.filter(ImageFilter.FIND_EDGES)
    return image
