import streamlit as st
from PIL import Image
import os
import uuid

from image_model import ImageRecognition
from nlp_model import ChatbotNLP

# Initialize models
image_recognition = ImageRecognition()
chatbot_nlp = ChatbotNLP()

# Streamlit UI
st.set_page_config(page_title="Conversational Image Recognition Chatbot", layout="centered")
st.title("🧠 Conversational Image Recognition Chatbot")

st.markdown("Upload an image, then ask a question about it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily to disk
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("uploads", temp_filename)
    os.makedirs("uploads", exist_ok=True)
    image.save(temp_path)

    with st.spinner("Recognizing objects in the image..."):
        predictions = image_recognition.predict(temp_path)
    st.success(f"Detected objects: {', '.join(predictions)}")

    # Set context for NLP model
    chatbot_nlp.set_image_context(", ".join(predictions))

    user_question = st.text_input("Ask a question about the image:")

    if user_question:
        with st.spinner("Generating response..."):
            response = chatbot_nlp.get_answer(user_question)
        st.markdown("### 🤖 Chatbot Response:")
        st.info(response)

    # Optional cleanup (delete image after processing)
    os.remove(temp_path)
