from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from src.vlm.vlm_base import VLM

class CLIP(VLM):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features
