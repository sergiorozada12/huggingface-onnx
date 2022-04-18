from src.utils import read_txt, save_txt
from src.config import NAME, NAME_ONNX, REGEX, MAX_LENGTH
from src.models import TranslationModelOnnx, TranslatorOnnx
from transformers import MarianTokenizer, MarianMTModel
import re
import numpy as np
import torch

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    model = TranslatorOnnx(name=NAME, split_regex=REGEX, max_length=MAX_LENGTH)
    text_translated = model.translate(TEXT)

    save_txt('data/test_text_translated_onnx.txt', text_translated)
