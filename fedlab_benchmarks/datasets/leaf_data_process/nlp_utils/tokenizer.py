"""
    This is modified by [RSE-Adversarial-Defense-Github]
    https://github.com/Raibows/RSE-Adversarial-Defense/tree/de7bb5afc94d3d262cf0b08f55952800161865ce
"""

import spacy
import re


class Tokenizer:

    def __init__(self, tokenizer_type='normal'):
        if tokenizer_type == 'normal':
            self.tokenizer = self.normal_token
        elif tokenizer_type == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
            self.tokenizer = self.spacy_token
        else:
            raise RuntimeError(f'Tokenizer type is error, do not have type {tokenizer_type}')
        self.token_type = tokenizer_type

    def pre_process(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"<br />", "", text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?,])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.strip()

    def normal_token(self, text: str, is_word=True) -> [str]:
        if is_word:
            return [tok for tok in text.split() if not tok.isspace()]
        else:
            return [tok for tok in text]

    def spacy_token(self, text: str, is_word=True) -> [str]:
        if is_word:
            text = self.nlp(text)
            return [token.text for token in text if not token.text.isspace()]
        else:
            return [tok for tok in text]

    def __call__(self, text: str, is_word=True) -> [str]:
        text = self.pre_process(text)
        words = self.tokenizer(text, is_word=is_word)
        return words
