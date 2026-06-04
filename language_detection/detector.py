import joblib
import os
from .preprocessor import TextPreprocessor
from .vectorizer import TfidfLangVectorizer
from .model import LanguageClassifier


MODEL_PATH      = os.path.join(os.path.dirname(__file__), "language_classifier.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

# =====================================================================================================


class LanguageDetector:
    def __init__(self):

        self.preprocessor = TextPreprocessor()
        self.model      = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

# =====================================================================================================


    def detect(self, text: str) -> str:

        # returns full language name e.g. "Arabic", "English"
        clean = self.preprocessor.clean_text(text)
        X     = self.vectorizer.transform([clean])
        pred  = self.model.predict(X)[0]

        lang_map = {
            "ar": "Arabic",    "bg": "Bulgarian", "de": "German",
            "el": "Modern Greek", "en": "English", "es": "Spanish",
            "fr": "French",    "hi": "Hindi",     "it": "Italian",
            "ja": "Japanese",  "nl": "Dutch",     "pl": "Polish",
            "pt": "Portuguese","ru": "Russian",   "sw": "Swahili",
            "th": "Thai",      "tr": "Turkish",   "ur": "Urdu",
            "vi": "Vietnamese","zh": "Chinese"
        }

        return lang_map.get(pred, pred)

# =====================================================================================================


    def detect_code(self, text: str) -> str:

        if len(text.strip()) < 15:
            return "unknown"

        clean = self.preprocessor.clean_text(text)

        import re
        clean = re.sub(r'(.)\1{2,}', r'\1\1', clean)

        if not clean.strip():
            return "unknown"

        X = self.vectorizer.transform([clean])

        try:
            proba = self.model.predict_proba(X)[0]
            max_p = proba.max()
            if max_p < 0.5:
                return "unknown"
            return self.model.classes_[proba.argmax()]

        except AttributeError:
            # sklearn version mismatch → fallback to predict only
            return self.model.predict(X)[0]