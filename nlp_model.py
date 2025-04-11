from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotNLP:
    def __init__(self):
        try:
            logger.info("Initializing ChatbotNLP model...")
            self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
            self.conversation_history = []
            self.current_image_context = None
            
            # Load a more conversational model for generating responses
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            # Fix the padding issue by setting padding_side to 'left'
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            logger.info("ChatbotNLP model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChatbotNLP: {str(e)}")
            raise

    def set_image_context(self, image_description):
        """Set the current image context and clear conversation history"""
        try:
            self.current_image_context = image_description
            self.conversation_history = []
            logger.info("Image context updated")
        except Exception as e:
            logger.error(f"Error setting image context: {str(e)}")
            raise
        
    def get_answer(self, question, context=None):
        try:
            if not self.current_image_context:
                return "I don't have any image context to work with. Please upload an image first."
            
            # Add the question to conversation history
            self.conversation_history.append(question)
            
            # First, try to answer using the QA model with image context
            qa_result = self.qa_model(
                question=question,
                context=self.current_image_context
            )
            
            # If the QA model is confident enough, use its answer
            if qa_result["score"] > 0.3:
                answer = qa_result["answer"]
                logger.info(f"Using QA model answer with confidence {qa_result['score']}")
            else:
                # Generate a conversational response
                answer = self._generate_conversational_response(question)
                logger.info("Using conversational model for response")
            
            # Add the answer to conversation history
            self.conversation_history.append(answer)
            
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."
    
    def _generate_conversational_response(self, question):
        try:
            # Prepare the conversation context
            conversation = " ".join(self.conversation_history[-3:])  # Use last 3 exchanges
            
            # Encode the conversation with proper padding
            inputs = self.tokenizer.encode(
                conversation + self.tokenizer.eos_token,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1000
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.chat_model.generate(
                    inputs,
                    max_length=1000,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=100,
                    top_p=0.7,
                    temperature=0.8
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response to only include the new part
            response = response[len(conversation):].strip()
            
            # If the response is empty or too short, provide a fallback
            if len(response) < 10:
                response = "I'm not sure about that specific detail, but " + self.current_image_context
            
            return response
        except Exception as e:
            logger.error(f"Error generating conversational response: {str(e)}")
            return "I apologize, but I had trouble generating a response. " + self.current_image_context
