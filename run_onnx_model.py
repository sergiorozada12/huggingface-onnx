from src.utils import read_txt, save_txt
from src.config import NAME, NAME_ONNX, REGEX
from src.models import TranslationEncoder, TranslationDecoder, TranslationModelOnnx
from transformers import MarianTokenizer, MarianMTModel
import re
import numpy as np
import torch

TEXT = read_txt('data/test_text.txt')

if __name__ == '__main__':
    encoder = TranslationEncoder()
    decoder = TranslationDecoder(50)

    tokenizer = MarianTokenizer.from_pretrained(NAME)

    text_filtered = TEXT.replace('\n', ' ').strip()
    sentences = re.split(REGEX, text_filtered)
    text_tokens = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt", max_length=50, padding='max_length')

    enc_inputs, enc_att_mask = text_tokens['input_ids'], text_tokens['attention_mask']
    hidden = encoder(enc_inputs, enc_att_mask)

    dec_inputs = [51969]*50
    dec_inputs[0] = 0
    dec_inputs = torch.tensor([dec_inputs], dtype=torch.int64)
    output = decoder(dec_inputs, hidden, enc_att_mask, 0)
    print(output.shape)

    model_cpu = MarianMTModel.from_pretrained(NAME)
    print(model_cpu.config.eos_token_id)
    print(model_cpu.config.pad_token_id)
    print(model_cpu.config.decoder_start_token_id)
    print(model_cpu.config.bos_token_id)

    model = TranslationModelOnnx(50, model_cpu.config)
    model.generate(text_tokens)
    print(model.max_decoder_length)

