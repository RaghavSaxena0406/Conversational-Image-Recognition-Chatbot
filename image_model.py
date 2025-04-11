import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRecognition:
    def __init__(self):
        try:
            logger.info("Initializing ImageRecognition model...")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            try:
                with open("classes.txt", "r") as f:
                    self.labels = json.load(f)
                logger.info(f"Loaded {len(self.labels)} classes from classes.txt")
            except Exception as e:
                logger.error(f"Error loading classes.txt: {str(e)}")
                raise
                
            logger.info("ImageRecognition model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ImageRecognition: {str(e)}")
            raise

    def predict(self, image_path):
        try:
            logger.info(f"Processing image: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                top5_prob, top5_indices = torch.topk(probabilities, 5)
                
                results = []
                for prob, idx in zip(top5_prob[0], top5_indices[0]):
                    idx_str = str(idx.item())
                    if idx_str in self.labels:
                        label = self.labels[idx_str]
                        confidence = float(prob.item())
                        if confidence > 0.1:
                            results.append({
                                "object": label,
                                "confidence": confidence
                            })
                    else:
                        logger.warning(f"Label not found for index {idx_str}")
                
                if not results:
                    for prob, idx in zip(top5_prob[0], top5_indices[0]):
                        idx_str = str(idx.item())
                        if idx_str in self.labels:
                            label = self.labels[idx_str]
                            confidence = float(prob.item())
                            results.append({
                                "object": label,
                                "confidence": confidence
                            })
                
                logger.info(f"Successfully processed image with {len(results)} predictions")
                return {
                    "predictions": results,
                    "description": self._generate_description(results)
                }
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def _generate_description(self, predictions):
        try:
            if not predictions:
                return "I couldn't identify any objects in the image."
                
            main_object = predictions[0]["object"]
            confidence = predictions[0]["confidence"]
            
            description = f"I can see a {main_object} in the image"
            if confidence < 0.8:
                description += f", though I'm not completely sure"
                
            if len(predictions) > 1:
                other_objects = [p["object"] for p in predictions[1:] if p["confidence"] > 0.1]
                if other_objects:
                    description += f". I also notice {', '.join(other_objects[:-1])}"
                    if len(other_objects) > 1:
                        description += f" and {other_objects[-1]}"
                    elif len(other_objects) == 1:
                        description += f" and {other_objects[0]}"
                        
            return description + "."
        except Exception as e:
            logger.error(f"Error generating description: {str(e)}")
            return "Error generating description for the image."
