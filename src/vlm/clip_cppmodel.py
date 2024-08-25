from clip_cpp import Clip
from PIL import Image 
from src.vlm.vlm_base import VLM

class CLIPCPP(VLM):
    def __init__(self, model_name="CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf"):
        super().__init__()
        self.model_name = model_name
        self.model = Clip(
            model_path_or_repo_id="gguf_models",
            model_file=self.model_name,
            verbosity=2,
        )

    def get_text_embedding(self, text):
        tokens = self.model.tokenize(text)
        text_features = self.model.encode_text(tokens)
        return text_features

    def get_image_embedding(self, image):
        image = Image.open(image)
        image_features = self.model.load_preprocess_encode_image(image)
        return image_features