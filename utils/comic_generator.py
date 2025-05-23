import re
import os
import nltk
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
import torch

nltk.download('punkt')

# Load model once
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    revision="fp16" if torch.cuda.is_available() else None,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Parse character: "dialogue"
def parse_script(script):
    lines = nltk.sent_tokenize(script)
    dialogue_pairs = []
    for line in lines:
        match = re.match(r"(\w+):\s*\"([^\"]+)\"", line)
        if match:
            speaker, dialogue = match.groups()
            dialogue_pairs.append((speaker, dialogue))
    return dialogue_pairs

# Draw speech bubble
def draw_bubble(img_pil, text, position, color="white"):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    bubble_w = int(w * 0.4)
    bubble_h = int(h * 0.15)
    x, y = position
    x1, y1 = x + bubble_w, y + bubble_h

    draw.rounded_rectangle([x, y, x1, y1], radius=15, fill=color)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()
    draw.text((x + 10, y + 10), text, fill="black", font=font)
    return img_pil

# Generate comic image
def generate_comic(script, prompt):
    dialogues = parse_script(script)
    base_image = pipe(prompt).images[0]

    positions = [
        (20, 20),
        (400, 20),
        (20, 300),
        (400, 300)
    ]
    img_pil = base_image.copy()
    for i, (speaker, text) in enumerate(dialogues):
        pos = positions[i % len(positions)]
        full_text = f"{speaker}: {text}"
        img_pil = draw_bubble(img_pil, full_text, pos)

    os.makedirs("output", exist_ok=True)
    output_path = "output/multi_speech_panel.png"
    img_pil.save(output_path)
    return output_path
