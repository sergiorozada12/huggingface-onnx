import re
from transformers import MarianMTModel, MarianTokenizer

class TranslationModel():
    def __init__(self, name, split_regex):
        self.split_regex = split_regex
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name)
    
    def _prepare_text(self, text):
        text_filtered = text.replace('\n', ' ').strip()
        return re.split(self.split_regex, text_filtered)

    def translate(self, text):
        sentences = self._prepare_text(text)
        batches = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")
        batches_translated = self.model.generate(**batches)
        sentences_translated = [self.tokenizer.decode(s, skip_special_tokens=True) for s in batches_translated]
        return " ".join(sentences_translated)
