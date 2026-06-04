import re
import string
import emoji


class TextPreprocessor:
    
    def __init__(self):
        self.punctuations = string.punctuation

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_links(self, text):
        return re.sub(r'http\S+|www\S+', '', text)

    def remove_emojis(self, text):
        return emoji.replace_emoji(text, replace='')

    def remove_punctuations(self, text):
        return text.translate(str.maketrans('', '', self.punctuations))

    def clean_text(self, text):
        text = text.lower()
        text = self.remove_numbers(text)
        text = self.remove_links(text)
        text = self.remove_emojis(text)
        text = self.remove_punctuations(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text