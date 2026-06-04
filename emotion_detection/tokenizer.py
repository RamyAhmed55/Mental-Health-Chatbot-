from transformers import DistilBertTokenizerFast


class EmotionTokenizer:
    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.tokenizer  = DistilBertTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )