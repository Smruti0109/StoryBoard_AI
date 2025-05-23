import re
import cv2
import numpy as np
import nltk
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import torch
import os

nltk.download('punkt')

# ========== Step 1: Load SD Model ==========
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# ========== Step 2: Script Input ==========
script = """
Boy: "Is anyone there?"
Girl: "Don't go into the forest!"
Boy: "But I heard something!"
"""

scene_description = "A teenage boy and a girl in a forest at night, manga style, black and white"

# ========== Step 3: Extract Dialogues ==========
def parse_script(script):
    lines = nltk.sent_tokenize(script)
    dialogue_pairs = []
    for line in lines:
        match = re.match(r"(\w+):\s*\"([^\"]+)\"", line)
        if match:
            speaker, dialogue = match.groups()
            dialogue_pairs.append((speaker, dialogue))
    return dialogue_pairs

dialogue_list = parse_script(script)

# ========== Step 4: Generate Scene Image ==========
print("Generating scene image...")
base_image = pipe(scene_description).images[0]

# ========== Step 5: Auto Bubble Placement ==========
def draw_bubble(img_pil, text, position, color="white"):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    bubble_w = int(w * 0.4)
    bubble_h = int(h * 0.15)
    x, y = position

    # Rectangle coords
    x0, y0 = x, y
    x1, y1 = x + bubble_w, y + bubble_h
    draw.rounded_rectangle([x0, y0, x1, y1], radius=15, fill=color)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    # Draw text
    text_x = x0 + 10
    text_y = y0 + 10
    draw.text((text_x, text_y), text, fill="black", font=font)

    return img_pil

# Convert to OpenCV for positioning
cv_img = np.array(base_image)
height, width, _ = cv_img.shape

# Predefined auto-placement positions (top-left, top-right, bottom-left, etc.)
positions = [
    (int(width * 0.05), int(height * 0.05)),
    (int(width * 0.55), int(height * 0.05)),
    (int(width * 0.05), int(height * 0.75)),
    (int(width * 0.55), int(height * 0.75)),
]

# ========== Step 6: Render Speech Bubbles ==========
img_pil = base_image.copy()
for i, (speaker, text) in enumerate(dialogue_list):
    pos = positions[i % len(positions)]
    full_text = f"{speaker}: {text}"
    img_pil = draw_bubble(img_pil, full_text, pos)

# ========== Step 7: Save Result ==========
os.makedirs("output", exist_ok=True)
img_pil.save("output/multi_speech_panel.png")
print("Saved as output/multi_speech_panel.png")
