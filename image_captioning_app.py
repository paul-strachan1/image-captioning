"""Flask web application for image captioning using BLIP model."""

import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


def load_model():
    """Load the pretrained processor and model."""
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def caption_image(input_image: np.ndarray, processor, model):
    """Generate a caption for the input image.

    Args:
        input_image (np.ndarray): Input image as numpy array
        processor: The BLIP processor
        model: The BLIP model

    Returns:
        str: Generated caption
    """
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image
    inputs = processor(raw_image, return_tensors="pt")

    # Generate a caption for the image
    out = model.generate(**inputs, max_length=50)

    # Decode the generated tokens to text
    return processor.decode(out[0], skip_special_tokens=True)


def main():
    """Main function to set up and launch the Gradio interface."""
    # Load model and processor
    processor, model = load_model()

    # Create the interface
    iface = gr.Interface(
        fn=lambda img: caption_image(img, processor, model),
        inputs=gr.Image(),
        outputs="text",
        title="Image Captioning",
        description="A web app for generating captions for images using a trained model."
    )

    # Launch the interface
    iface.launch()


if __name__ == "__main__":
    main()
