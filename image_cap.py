"""Basic image captioning script using BLIP model."""

from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


def load_model():
    """Load the pretrained processor and model."""
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def generate_caption(image_path, processor, model):
    """Generate a caption for the given image.
    
    Args:
        image_path (str): Path to the image file
        processor: The BLIP processor
        model: The BLIP model
        
    Returns:
        str: Generated caption
    """
    # Load and convert image to RGB format
    image = Image.open(image_path).convert('RGB')
    # Process the image
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    # Generate caption
    outputs = model.generate(**inputs, max_length=50)
    # Decode and return caption
    return processor.decode(outputs[0], skip_special_tokens=True)


def main():
    """Main function to run the image captioning."""
    # Load model and processor
    processor, model = load_model()
    # Set image path
    img_path = "example_pic.jpg"
    # Generate and print caption
    caption = generate_caption(img_path, processor, model)
    print(caption)


if __name__ == "__main__":
    main()
