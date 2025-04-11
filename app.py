from flask import Flask, request, jsonify
from flask_cors import CORS
from image_model import ImageRecognition
from nlp_model import ChatbotNLP
import os

app = Flask(__name__)
CORS(app)

# Ensure upload directory exists
os.makedirs('uploads', exist_ok=True)

image_recognition = ImageRecognition()
chatbot_nlp = ChatbotNLP()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    # Save the file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Get image analysis
    try:
        analysis = image_recognition.predict(file_path)
        
        # Set the image context for the chatbot
        chatbot_nlp.set_image_context(analysis['description'])
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "message": "Image uploaded and analyzed successfully"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
        
    question = data["question"]
    try:
        answer = chatbot_nlp.get_answer(question)
        return jsonify({
            "success": True,
            "answer": answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
