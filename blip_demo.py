"""
BLIP Image Captioning Demo & Utilities
-------------------------------------
- Load the BLIP model and processor (Salesforce/blip-image-captioning-base)
- Generate captions for local or online images
- Includes fallback to a placeholder image if no image is found
"""

import os
import torch
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_model(model_name="Salesforce/blip-image-captioning-base"):
    """
    Load BLIP model + processor.
    Uses local cache if available, otherwise downloads from Hugging Face.

    Returns:
        processor (BlipProcessor): Preprocessing processor for BLIP
        model (BlipForConditionalGeneration): BLIP model
    """
    print(f"🔍 Loading model: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    print("✅ Model ready.")
    return processor, model


def generate_caption(image: Image.Image, processor, model, max_tokens: int = 30) -> str:
    """
    Generate a caption for a given PIL image using BLIP.

    Args:
        image (PIL.Image.Image): Input image
        processor: BLIP processor
        model: BLIP model
        max_tokens (int): Maximum number of tokens to generate

    Returns:
        str: Generated caption
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=max_tokens)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def load_image(img_path: str) -> Image.Image:
    """
    Load an image from a local path or URL. If loading fails, use a placeholder image.

    Args:
        img_path (str): Local path or online URL

    Returns:
        PIL.Image.Image: Loaded image
    """
    try:
        if img_path.startswith("http://") or img_path.startswith("https://"):
            image = Image.open(requests.get(img_path, stream=True).raw).convert("RGB")
            print("🌐 Loaded online image.")
        else:
            image = Image.open(img_path).convert("RGB")
            print(f"💾 Loaded local image: {img_path}")
    except Exception as e:
        print(f"⚠️ Failed to load image ({e}). Using placeholder.")
        placeholder_path = os.path.join(os.path.dirname(__file__), "placeholder.jpg")
        if not os.path.exists(placeholder_path):
            Image.new("RGB", (224, 224), color="white").save(placeholder_path)
            print("Created placeholder image.")
        image = Image.open(placeholder_path)
    return image


def main():
    print("🚀 Running BLIP Image Captioning Demo...")

    # Load model and processor
    processor, model = load_model()

    # Example image path or URL
    img_path = "image/Cat03.jpg"  # replace with your local image or URL
    image = load_image(img_path)

    # Generate caption
    caption = generate_caption(image, processor, model)
    print(f"✅ Demo caption: {caption}")


if __name__ == "__main__":
    main()
