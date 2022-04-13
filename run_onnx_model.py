from src.utils import read_txt, save_txt
from src.config import NAME, NAME_ONNX, REGEX, MAX_LENGTH
from src.models import TranslationModelOnnx
from transformers import MarianTokenizer, MarianMTModel
import re
import numpy as np
import torch

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    tokenizer = MarianTokenizer.from_pretrained(NAME)
    text_filtered = TEXT.replace('\n', ' ').strip()
    sentences = re.split(REGEX, text_filtered)
    text_tokens = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt", max_length=50, padding='max_length')

    config_file = MarianMTModel.from_pretrained(NAME).config
    model = TranslationModelOnnx(MAX_LENGTH, config_file)
    print(model.generate(text_tokens))
