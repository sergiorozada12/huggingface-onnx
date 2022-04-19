import re
import time

import numpy as np

from transformers import MarianTokenizer, MarianMTModel

from src.utils import read_txt
from src.config import NAME, REGEX
from src.models import TranslationModelOnnx


TEXT = read_txt('data/test_text.txt')

tokenizer = MarianTokenizer.from_pretrained(NAME)
model_torch = MarianMTModel.from_pretrained(NAME)
model_onnx = TranslationModelOnnx(model_torch.config)

if __name__ == '__main__':
    sentences = re.split(REGEX, TEXT)
    batches = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")

    torch_times, onnx_times = [], []

    for _ in range(10):
        model_torch.generate(**batches)
        model_onnx.generate(batches)

    for _ in range(10):
        start = time.time()
        _ = model_torch.generate(
            **batches,
            num_beams=1,
            num_beam_groups=1,
            do_sample=False,
            constraints=None,
            force_words_ids=None,
        )
        end = time.time()
        torch_times.append(1000*(end - start))

    for _ in range(10):
        start = time.time()
        _ = model_onnx.generate(batches)
        end = time.time()
        onnx_times.append(1000*(end - start))

    print(f"Torch latency: {np.around(np.mean(torch_times), 2)} ms")
    print(f"ONNX latency: {np.around(np.mean(onnx_times), 2)} ms")
