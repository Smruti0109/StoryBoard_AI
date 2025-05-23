import streamlit as st
from utils.comic_generator import generate_comic

st.set_page_config(page_title="Comic Generator AI", layout="wide")

st.title("🎨 AI-Powered Comic/Manga Panel Generator")
st.markdown("Upload a script and generate a manga-style image with auto-placed speech bubbles!")

script = st.text_area("📜 Enter Script (format: Character: \"dialogue\")", height=200, value="""
Boy: "Is anyone there?"
Girl: "Don't go into the forest!"
Boy: "But I heard something!"
""")

prompt = st.text_input("🎭 Scene Description (for AI Image)", value="A teenage boy and a girl in a dark forest, manga style, black and white")

if st.button("✨ Generate Comic Panel"):
    with st.spinner("Generating..."):
        output_image = generate_comic(script, prompt)
        st.image(output_image, caption="🖼️ Your Generated Comic Panel", use_column_width=True)
