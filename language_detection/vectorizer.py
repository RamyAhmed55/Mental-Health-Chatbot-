from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfLangVectorizer:
    def __init__(self, max_features=5000, ngram_range=(2, 5)):
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            max_features=max_features,
            ngram_range=ngram_range
        )
        self.fitted = False

    def fit_transform(self, texts):
        X = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return X

    def transform(self, texts):
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        return self.vectorizer.transform(texts)