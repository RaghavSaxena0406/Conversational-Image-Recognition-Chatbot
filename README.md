# Conversational Image Recognition Chatbot

## Project Overview
This project implements an intelligent chatbot that can recognize objects in images and engage in natural conversations about them. The system combines computer vision and natural language processing to provide a seamless conversational experience about image content.

### Key Features
- Image object detection and recognition
- Natural language understanding and response generation
- Conversational context maintenance
- Detailed image descriptions
- Confidence-based predictions
- Multiple object detection in single images

### Technical Stack
- **Backend Framework**: Flask
- **Image Recognition**: PyTorch with ResNet50
- **Natural Language Processing**: Transformers (RoBERTa and DialoGPT)
- **Image Processing**: PIL (Python Imaging Library)
- **API Communication**: Flask-CORS

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate

   # On Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; import flask; import transformers; print('All dependencies installed successfully!')"
   ```

## Project Structure
```
project_root/
├── app.py                  # Main Flask application
├── image_model.py          # Image recognition model
├── nlp_model.py            # Natural language processing model
├── classes.txt             # Image class labels
├── requirements.txt        # Project dependencies
├── uploads/                # Directory for uploaded images
├── test_images/           # Directory for test images
└── README.md              # Project documentation
```

## Running the Application

1. **Start the Flask Server**
   ```bash
   # Make sure virtual environment is activated
   python app.py
   ```
   The server will start at `http://127.0.0.1:5000/`

2. **Test the Application**
   ```bash
   # In a new terminal with activated virtual environment
   python test_api.py
   ```

## API Endpoints

1. **Upload Image**
   - **Endpoint**: `/upload`
   - **Method**: POST
   - **Content-Type**: multipart/form-data
   - **Parameters**: 
     - `image`: Image file to upload
   - **Response**: JSON with image analysis results

2. **Ask Questions**
   - **Endpoint**: `/ask`
   - **Method**: POST
   - **Content-Type**: application/json
   - **Parameters**:
     - `question`: Question about the uploaded image
   - **Response**: JSON with answer

## Example Usage

1. **Upload an Image**
   ```python
   import requests

   url = "http://127.0.0.1:5000/upload"
   files = {"image": open("path/to/image.jpg", "rb")}
   response = requests.post(url, files=files)
   print(response.json())
   ```

2. **Ask Questions**
   ```python
   url = "http://127.0.0.1:5000/ask"
   data = {"question": "What is in the image?"}
   response = requests.post(url, json=data)
   print(response.json())
   ```

## Troubleshooting

1. **Port Already in Use**
   - Error: "Address already in use"
   - Solution: Change the port in app.py or kill the process using the port

2. **Model Loading Issues**
   - Error: "Error initializing ImageRecognition"
   - Solution: Check internet connection and try reinstalling dependencies

3. **Memory Issues**
   - Error: "CUDA out of memory"
   - Solution: Close other applications or reduce batch size

4. **File Permission Issues**
   - Error: "Permission denied"
   - Solution: Check directory permissions and ensure write access

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments
- PyTorch team for the ResNet50 model
- Hugging Face for the Transformers library
- Flask team for the web framework 