# Image Captioning Application

This project contains tools for generating captions for images using AI models.

## Features

- Image captioning using pre-trained models
- URL-based image captioning
- Automated caption generation

## Project Structure

```
gen_ai_apps/
├── image_caption/
│   ├── image_cap.py           # Basic image captioning
│   ├── image_captioning_app.py # Flask web application
│   ├── automate_url_captioner.py # URL-based captioning
│   └── captions.txt           # Sample captions
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <[repository-url](https://github.com/paul-strachan1/image-captioning)>
   cd gen_ai_apps
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Image Captioning
```bash
python image_caption/image_cap.py
```

### Web Application
```bash
python image_caption/image_captioning_app.py
```

### URL-based Captioning
```bash
python image_caption/automate_url_captioner.py
```

## Requirements

- Python 3.7+
- See requirements.txt for detailed dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
