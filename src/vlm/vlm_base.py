from abc import ABC, abstractmethod

class VLM(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def get_text_embedding(self, text):
        pass

    @abstractmethod
    def get_image_embedding(self, image):
        pass
