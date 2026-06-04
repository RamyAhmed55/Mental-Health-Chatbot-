from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class LanguageClassifier:

    def __init__(self):

        self.model = LogisticRegression(max_iter=200, solver="lbfgs")
        self.fitted = False

        self.lang_map = {
            "ar": "Arabic",    "bg": "Bulgarian", "de": "German",
            "el": "Modern Greek", "en": "English", "es": "Spanish",
            "fr": "French",    "hi": "Hindi",     "it": "Italian",
            "ja": "Japanese",  "nl": "Dutch",     "pl": "Polish",
            "pt": "Portuguese","ru": "Russian",   "sw": "Swahili",
            "th": "Thai",      "tr": "Turkish",   "ur": "Urdu",
            "vi": "Vietnamese","zh": "Chinese"
        }

# =====================================================================================================

    def train(self, X, y, test_size=0.2, random_state=42):
        
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size, stratify=y, random_state=random_state)
        
        self.model.fit(X_train, y_train)
        self.fitted = True
        y_pred = self.model.predict(X_test)


        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        return X_train, X_test, y_train, y_test

# =====================================================================================================


    def predict(self, texts, vectorizer):

        if not self.fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        X_new = vectorizer.transform(texts)
        preds = self.model.predict(X_new)
        return [self.lang_map[label] for label in preds]