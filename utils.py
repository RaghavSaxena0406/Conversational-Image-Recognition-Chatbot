import os

UPLOAD_FOLDER = "uploads"

def save_uploaded_file(file):
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    return file_path

def generate_context(label):
    # Basic context generation based on the label
    context_dict = {
        "dog": "Dogs are loyal pets and come in many breeds.",
        "cat": "Cats are independent animals known for their agility.",
        "car": "Cars are vehicles used for transportation.",
        "tree": "Trees provide oxygen and are essential for the environment."
    }
    return context_dict.get(label, "I see something, but I need more details.")
