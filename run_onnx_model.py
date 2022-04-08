from src.utils import read_txt, save_txt
from src.config import NAME, NAME_ONNX, REGEX
from src.models import TranslationEncoder, TranslationDecoder
from transformers import MarianTokenizer
import re
import numpy as np
import torch

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    encoder = TranslationEncoder()
    decoder = TranslationDecoder()

    tokenizer = MarianTokenizer.from_pretrained(NAME)

    text_filtered = TEXT.replace('\n', ' ').strip()
    sentences = re.split(REGEX, text_filtered)
    text_tokens = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt", max_length=50, padding='max_length')

    enc_inputs, enc_att_mask = text_tokens['input_ids'], text_tokens['attention_mask']
    hidden = encoder(enc_inputs, enc_att_mask)[0]
    hidden = torch.tensor(hidden, dtype=torch.float32)

    index = 0
    dec_att_mask = [1]*(index + 1) + [0]*(50 - index - 1)
    dec_att_mask = torch.tensor([dec_att_mask], dtype=torch.int64)

    dec_inputs = [51969]*50
    dec_inputs[0] = 0
    dec_inputs = torch.tensor([dec_inputs], dtype=torch.int64)
    output = decoder(dec_inputs, dec_att_mask, hidden, enc_att_mask, index)
