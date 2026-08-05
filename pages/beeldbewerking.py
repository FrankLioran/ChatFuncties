# pages/beeldbewerking.py
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io
from pathlib import Path

# We importeren alleen wat daadwerkelijk bestaat in image.py
from image import generate_and_save_image

st.set_page_config(page_title="Eva — Beeldbewerking", layout="wide")

st.title("🎨 Eva — Beeldbewerking & Beeldgeneratie")
st.caption("Bewerk je eigen afbeeldingen of genereer nieuwe beelden via de cloud.")

# ---------------------------------------------------------
# TAB 1: EIGEN AFBEELDING BEWERKEN (PIL)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📷 Afbeelding Bewerken", "🎨 Nieuwe Afbeelding Genereren"])

with tab1:
    uploaded = st.file_uploader("Upload een afbeelding", type=["png", "jpg", "jpeg"], key="editor_uploader")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col_orig, col_edit = st.columns(2)
        with col_orig:
            st.image(img, caption="Origineel", use_container_width=True)

        st.markdown("### ✨ Bewerkingen")

        c1, c2 = st.columns(2)
        with c1:
            brightness = st.slider("Helderheid", 0.1, 3.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
            saturation = st.slider("Kleurverzadiging", 0.1, 3.0, 1.0, 0.1)
            sharpness = st.slider("Scherpte", 0.1, 3.0, 1.0, 0.1)

        with c2:
            rotate_deg = st.slider("Rotatie (graden)", -180, 180, 0)
            flip_dir = st.selectbox("Spiegelen", ["Geen", "Horizontaal", "Verticaal"])
            effect = st.selectbox(
                "Effect",
                ["Geen", "Zwart-Wit", "Sepia", "Vervaag (Blur)", "Scherpe Schets (Detail)"]
            )

        # Pas bewerkingen toe
        edited = img.copy()

        # Enhancements
        if brightness != 1.0:
            edited = ImageEnhance.Brightness(edited).enhance(brightness)
        if contrast != 1.0:
            edited = ImageEnhance.Contrast(edited).enhance(contrast)
        if saturation != 1.0:
            edited = ImageEnhance.Color(edited).enhance(saturation)
        if sharpness != 1.0:
            edited = ImageEnhance.Sharpness(edited).enhance(sharpness)

        # Rotatie & Spiegelen
        if rotate_deg != 0:
            edited = edited.rotate(-rotate_deg, expand=True)
        if flip_dir == "Horizontaal":
            edited = edited.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_dir == "Verticaal":
            edited = edited.transpose(Image.FLIP_TOP_BOTTOM)

        # Effecten
        if effect == "Zwart-Wit":
            edited = edited.convert("L").convert("RGB")
        elif effect == "Vervaag (Blur)":
            edited = edited.filter(ImageFilter.BLUR)
        elif effect == "Scherpe Schets (Detail)":
            edited = edited.filter(ImageFilter.DETAIL)
        elif effect == "Sepia":
            # Snelle Sepia filter via matrix
            sepia_matrix = (
                0.393, 0.769, 0.189, 0,
                0.349, 0.686, 0.168, 0,
                0.272, 0.534, 0.131, 0
            )
            edited = edited.convert("RGB", sepia_matrix)

        with col_edit:
            st.image(edited, caption="Bewerkt resultaat", use_container_width=True)

        # Downloadknop
        buf = io.BytesIO()
        edited.save(buf, format="PNG")
        st.download_button(
            "💾 Download bewerkte afbeelding",
            data=buf.getvalue(),
            file_name="eva_bewerkt.png",
            mime="image/png",
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 2: POLLINATIONS CLOUD GENERATOR
# ---------------------------------------------------------
with tab2:
    st.header("🎨 Pollinations Cloud Generator")
    st.write("Genereer vanuit de cloud gloednieuwe afbeeldingen op basis van tekst.")

    prompt_text = st.text_input("Wat wil je maken?", value="a breathtaking landscape of misty mountains at sunrise, cinematic lighting")

    col_style, col_ar = st.columns(2)
    with col_style:
        style_preset = st.selectbox("Stijl toevoegen", ["Geen", "Photorealistic", "Anime", "Digital Art", "Cinematic", "Cyberpunk"])
    with col_ar:
        ar_preset = st.selectbox("Formaat", ["Vierkant (1:1)", "Liggend (16:9)", "Staand (9:16)"])

    if st.button("✨ Genereer Afbeelding", key="btn_generate_page"):
        # Bouw de prompt op met de gekozen presets
        final_prompt = prompt_text
        if style_preset != "Geen":
            final_prompt += f", {style_preset} style"

        if ar_preset == "Liggend (16:9)":
            final_prompt += ", widescreen 16:9"
        elif ar_preset == "Staand (9:16)":
            final_prompt += ", portrait 9:16"

        with st.spinner("Eva weeft je afbeelding in de cloud..."):
            image_path = generate_and_save_image(final_prompt)

            if image_path and Path(image_path).exists():
                st.image(str(image_path), caption=f"Resultaat voor: '{final_prompt}'", use_container_width=True)

                with open(image_path, "rb") as f:
                    st.download_button(
                        "📥 Download deze afbeelding",
                        data=f.read(),
                        file_name="pollinations_creatie.png",
                        mime="image/png"
                    )
            else:
                st.error("Het genereren is helaas niet gelukt. Probeer het over een minuutje nog eens.")
