import requests
import os
from PIL import Image
import io

def download_image(url, save_path):
    """Download an image from URL and save it locally"""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

def test_upload_endpoint(image_path):
    """Test the upload endpoint"""
    print(f"\nTesting upload endpoint with image: {image_path}")
    url = "http://127.0.0.1:5000/upload"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.json()
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return None

def test_ask_endpoint(question):
    """Test the ask endpoint"""
    print(f"\nTesting ask endpoint with question: {question}")
    url = "http://127.0.0.1:5000/ask"
    
    try:
        data = {"question": question}
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.json()
    except Exception as e:
        print(f"Error during ask: {str(e)}")
        return None

def main():
    # Create test images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Sample image URLs (using publicly available images)
    test_images = [
        {
            'url': 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg',
            'name': 'dog.jpg',
            'description': 'A dog image'
        },
        {
            'url': 'https://raw.githubusercontent.com/pytorch/hub/master/images/cat.jpg',
            'name': 'cat.jpg',
            'description': 'A cat image'
        }
    ]
    
    # Download and test each image
    for img in test_images:
        image_path = os.path.join('test_images', img['name'])
        
        # Download image
        print(f"\nDownloading {img['description']}...")
        if download_image(img['url'], image_path):
            print(f"Successfully downloaded {img['name']}")
            
            # Test upload endpoint
            upload_response = test_upload_endpoint(image_path)
            
            if upload_response and upload_response.get('success'):
                # Test various questions
                questions = [
                    "What is in the image?",
                    "Can you describe what you see?",
                    "What are the main objects in this image?",
                    "Is there an animal in the image?"
                ]
                
                for question in questions:
                    test_ask_endpoint(question)
        else:
            print(f"Failed to download {img['name']}")

if __name__ == "__main__":
    main() 