import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
import uuid
import random
import requests
import json
import subprocess
import os 
from pathlib import Path
from config import DOCUMENT_FOLDER, DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE # Projectmodules
from chat import answer_question, load_profile, get_persona_image
from documents import (
    scan_and_index_folder_full,
    scan_and_index_folder_lazy,
)

from image import (
    start_comfyui,
    generate_pollinations,
    generate_pollinations_styled, 
    save_pollinations_image,
    start_paintnet,
    generate_comfy_image,
    image_editor_ui,

    # Pollinations geavanceerde functies
    build_pollinations_url,
    fetch_pollinations_image,
    save_to_comfyui_input,
    generate_pollinations_image,
    upload_image_for_pollinations,

    # Stijl presets
    poll_style_flux,
    poll_style_sdxl,
    poll_style_anime,
    poll_style_realistic,
    poll_style_cinematic,

    # Aspect ratio presets
    poll_ar_square,
    poll_ar_wide,
    poll_ar_vertical,

    # Negative prompt presets
    poll_negative_no_text,
    poll_negative_no_blur,

    # Seed presets
    poll_seed_random,
    poll_seed_fixed,

    # Steps presets
    poll_steps_fast,
    poll_steps_quality,

    # Image-to-image
    poll_image_to_image,

    # Video workflow
    comfyui_generate_video_from_image
)

from image import (
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_sharpness,
    apply_rotate,
    apply_flip,
    apply_sepia,
    apply_effect
)

st.set_page_config(page_title="Eva — Beeldbewerking", layout="wide")


st.title("🎨 Eva — Beeldbewerking, beeldgeneratie")

# ---------------------------------------------------------
# ComfyUI en Paintnet starten, afbeelding uploaden
# ---------------------------------------------------------

st.header("ComfyUI starten")
if st.button("Start ComfyUI"):
    st.success(start_comfyui())
 
st.header("🎨 Paint.NET openen")
if st.button("Start Paint.NET"):
        st.success(start_paintnet())

uploaded = st.file_uploader("📷 Upload een afbeelding", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGBA")
    st.image(img, caption="Origineel", use_container_width=True)

    # ---------------------------------------------------------
    # Pollinations — Geavanceerde Generator
    # ---------------------------------------------------------

    st.header("🎨 Pollinations Generator")

    positive_prompt = st.text_input("Positive prompt", "a red rose in the rain")
    negative_prompt = st.text_input("Negative prompt (optioneel)", "text, watermark, blurry, distorted")

    # Basisgeneratie
    if st.button("✨ Genereer basisafbeelding"):
        path, img = generate_pollinations_styled(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Afbeelding opgeslagen: {path}")
        else:
            st.error("Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Stijl presets
    # ---------------------------------------------------------
    st.subheader("🎨 Stijl")

    if st.button("Flux"):
        path, img = poll_style_flux(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Flux stijl opgeslagen: {path}")
        else:
            st.error("Flux: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("SDXL"):
        path, img = poll_style_sdxl(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"SDXL stijl opgeslagen: {path}")
        else:
            st.error("SDXL: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("Anime"):
        path, img = poll_style_anime(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Anime stijl opgeslagen: {path}")
        else:
            st.error("Anime: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("Realistic"):
        path, img = poll_style_realistic(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Realistic stijl opgeslagen: {path}")
        else:
            st.error("Realistic: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("Cinematic"):
        path, img = poll_style_cinematic(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Cinematic stijl opgeslagen: {path}")
        else:
            st.error("Cinematic: Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Aspect ratio presets
    # ---------------------------------------------------------
    st.subheader("🖼️ Aspect Ratio")

    if st.button("1:1"):
        path, img = poll_ar_square(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"1:1 opgeslagen: {path}")
        else:
            st.error("1:1: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("16:9"):
        path, img = poll_ar_wide(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"16:9 opgeslagen: {path}")
        else:
            st.error("16:9: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("9:16"):
        path, img = poll_ar_vertical(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"9:16 opgeslagen: {path}")
        else:
            st.error("9:16: Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Seed presets
    # ---------------------------------------------------------
    st.subheader("🎲 Seed")

    if st.button("Random seed"):
        path, img = poll_seed_random(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Random seed opgeslagen: {path}")
        else:
            st.error("Random seed: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("Seed 12345"):
        path, img = poll_seed_fixed(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Seed 12345 opgeslagen: {path}")
        else:
            st.error("Seed 12345: Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Negative prompt presets
    # ---------------------------------------------------------
    st.subheader("🧪 Negative Prompt")

    if st.button("No text"):
        path, img = poll_negative_no_text(positive_prompt)
        if img:
            st.image(img)
            st.success(f"No text opgeslagen: {path}")
        else:
            st.error("No text: Pollinations gaf geen geldige afbeelding terug.")

    if st.button("No blur"):
        path, img = poll_negative_no_blur(positive_prompt)
        if img:
            st.image(img)
            st.success(f"No blur opgeslagen: {path}")
        else:
            st.error("No blur: Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Steps presets
    # ---------------------------------------------------------
    st.subheader("⚙️ Steps")

    if st.button("Fast (10)"):
        path, img = poll_steps_fast(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Fast (10) opgeslagen: {path}")
        else:
            st.error("Fast (10): Pollinations gaf geen geldige afbeelding terug.")

    if st.button("Quality (40)"):
        path, img = poll_steps_quality(positive_prompt, negative_prompt)
        if img:
            st.image(img)
            st.success(f"Quality (40) opgeslagen: {path}")
        else:
            st.error("Quality (40): Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Image-to-image
    # ---------------------------------------------------------
    st.subheader("🖼️ Image-to-Image")

    uploaded_img = st.file_uploader("Upload afbeelding voor I2I", type=["png", "jpg", "jpeg"])

    if uploaded_img and st.button("Genereer I2I"):
        img_bytes = uploaded_img.read()
        path, img = poll_image_to_image(positive_prompt, negative_prompt, img_bytes)
        if img:
            st.image(img)
            st.success(f"Image-to-image opgeslagen: {path}")
        else:
            st.error("Image-to-image: Pollinations gaf geen geldige afbeelding terug.")

    # ---------------------------------------------------------
    # Paint Advanced
    # ---------------------------------------------------------

    st.markdown("## ✨ Basisbewerkingen")

    brightness = st.slider("Helderheid", 0.1, 3.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
    saturation = st.slider("Kleurverzadiging", 0.1, 3.0, 1.0, 0.1)
    sharpness = st.slider("Scherpte", 0.1, 3.0, 1.0, 0.1)

    st.markdown("## 🔄 Rotatie & Spiegeling")

    rotate_deg = st.slider("Rotatie (graden)", -180, 180, 0)
    flip_dir = st.selectbox("Spiegelen", ["Geen", "Horizontaal", "Verticaal"])

    st.markdown("## 🎨 Effecten")

    effect = st.selectbox(
        "Effect",
        ["Geen", "Zwart-Wit", "Sepia", "Gaussiaanse Vervaag (Blur)", "Scherpe Schets (Detail)"]
    )

    if st.button("✨ Pas bewerkingen toe"):
        edited = img.copy()

        edited = apply_brightness(edited, brightness)
        edited = apply_contrast(edited, contrast)
        edited = apply_saturation(edited, saturation)
        edited = apply_sharpness(edited, sharpness)

        if rotate_deg != 0:
            edited = apply_rotate(edited, rotate_deg)

        if flip_dir != "Geen":
            edited = apply_flip(edited, flip_dir)

        if effect != "Geen":
            edited = apply_effect(edited, effect)

        st.image(edited, caption="Bewerkt resultaat", use_container_width=True)

        buf = io.BytesIO()
        edited.save(buf, format="PNG")
        st.download_button(
            "💾 Download bewerkte afbeelding",
            data=buf.getvalue(),
            file_name="eva_bewerkt.png",
            mime="image/png"
        )
