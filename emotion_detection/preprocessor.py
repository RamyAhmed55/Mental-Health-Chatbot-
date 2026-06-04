import re
import string
import emoji


class EmotionPreprocessor:

    def __init__(self):
        self.punctuations = string.punctuation
        self.emoji_map = {
            "😀": "joy_emoji",  "😂": "joy_emoji", "🤣": "joy_emoji",
            "😆": "joy_emoji",  "😁": "joy_emoji",
            "😥": "sadness_emoji", "☹": "sadness_emoji",
            "😡": "anger_emoji",   "🤬": "anger_emoji",
            "😱": "fear_emoji",    "😰": "fear_emoji", "😨": "fear_emoji",
            "😯": "surprise_emoji",
            "😍": "love_emoji",    "😘": "love_emoji", "🥰": "love_emoji"}

# =====================================================================================================


    def normalize_emojis(self, text):
        return emoji.emojize(emoji.demojize(text), language='en')
    
# =====================================================================================================


    def replace_emojis(self, text):
        for emo, token in self.emoji_map.items():
            text = text.replace(emo, f" {token} ")
        text = emoji.replace_emoji(text, replace='')
        return text

# =====================================================================================================


    def clean_text(self, text):

        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = self.normalize_emojis(text)
        text = self.replace_emojis(text)
        text = text.translate(str.maketrans('', '', self.punctuations))
        text = re.sub(r'\s+', ' ', text).strip()
        return text