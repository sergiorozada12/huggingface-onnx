import re
import time

import numpy as np

import torch
from transformers import MarianTokenizer, MarianMTModel

from src.utils import read_txt
from src.config import NAME, REGEX
from src.models import TranslationModelOnnx
from src.wrappers import MarianDecoderWrapped


TEXT = read_txt('data/test_text.txt')

tokenizer = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME)

if __name__ == '__main__':
    sentences = re.split(REGEX, TEXT)
    batches = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")

    input_ids = batches['input_ids']
    attention_mask = batches['attention_mask']

    encoder_hidden_states = model.model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )[0]

    decoder = model.model.decoder
    decoder.__class__ = MarianDecoderWrapped

    decoder_input_ids = torch.randint(10_000, (input_ids.shape[0], 1), requires_grad=False)
    decoder_input_ids[:, 0] = model.config.decoder_start_token_id
    output = decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=attention_mask
    )

    print(output)
