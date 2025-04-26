"""Script to automatically generate captions for images from a webpage."""

from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration


def load_model():
    """Load the pretrained processor and model."""
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


def process_image(img_url, processor, model):
    """Process a single image and generate its caption.

    Args:
        img_url (str): URL of the image to process
        processor: The BLIP processor
        model: The BLIP model

    Returns:
        str: Generated caption or None if processing fails
    """
    try:
        # Download the image
        response = requests.get(img_url, timeout=10)
        # Convert the image data to a PIL Image
        raw_image = Image.open(BytesIO(response.content))

        # Skip very small images
        if raw_image.size[0] * raw_image.size[1] < 400:
            return None

        raw_image = raw_image.convert('RGB')

        # Process the image
        inputs = processor(raw_image, return_tensors="pt")
        # Generate a caption for the image
        out = model.generate(**inputs, max_new_tokens=50)
        # Decode the generated tokens to text
        return processor.decode(out[0], skip_special_tokens=True)
    except (requests.RequestException, Image.UnidentifiedImageError) as e:
        print(f"Error processing image {img_url}: {e}")
        return None


def main():
    """Main function to scrape images and generate captions."""
    # Load model and processor
    processor, model = load_model()

    # URL of the page to scrape
    url = "https://www.bbc.co.uk/news"

    # Download and parse the page
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all img elements
    img_elements = soup.find_all('img')

    # Open a file to write the captions
    with open("captions.txt", "w", encoding="utf-8") as caption_file:
        # Iterate over each img element
        for img_element in img_elements:
            img_url = img_element.get('src')

            # Skip if the image is an SVG or too small (likely an icon)
            if 'svg' in img_url or '1x1' in img_url:
                continue

            # Correct the URL if it's malformed
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            elif not img_url.startswith(('http://', 'https://')):
                continue

            # Process the image and get caption
            caption = process_image(img_url, processor, model)
            if caption:
                # Write the caption to the file
                caption_file.write(f"{img_url}: {caption}\n")


if __name__ == "__main__":
    main()
