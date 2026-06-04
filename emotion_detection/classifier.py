import torch
import os
from .preprocessor import EmotionPreprocessor
from .tokenizer import EmotionTokenizer
from .model import EmotionClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.pt")

# class weights from training — fixed values, no need to recompute
CLASS_WEIGHTS = [0.5715, 0.4973, 2.0449, 1.2351, 1.3766, 4.6620]


class EmotionDetector:
    def __init__(self):

        self.preprocessor = EmotionPreprocessor()
        self.tokenizer    = EmotionTokenizer()
        self.model        = EmotionClassifier(num_classes=6,class_weights=CLASS_WEIGHTS)
        self.model.load_state_dict( torch.load(MODEL_PATH, map_location="cpu", weights_only=True))

        self.model.eval()

# =====================================================================================================

    def classify(self, text: str) -> str:
        
        # returns emotion string or "normal" if below threshold
        # returns "unknown" if language is not English

        clean     = self.preprocessor.clean_text(text)
        encodings = self.tokenizer.tokenize([clean])
        result    = self.model.predict(
            encodings["input_ids"],
            encodings["attention_mask"]
        )
        return result[0]