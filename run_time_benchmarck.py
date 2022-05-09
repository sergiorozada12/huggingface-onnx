import re
import time

import numpy as np

from transformers import MarianTokenizer, MarianMTModel

from src.utils import read_txt
from src.config import NAME, REGEX, ITERATIONS
from src.models import TranslationModelOnnx


TEXT = read_txt('data/test_text.txt')

def warm_up(model_torch, model_onnx, batches):
    for _ in range(10):
        model_torch.generate(**batches)
        model_onnx.generate(batches)

def measure_time_torch(model, batches, iterations):
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model.generate(
            **batches,
            num_beams=1,
            num_beam_groups=1,
            do_sample=False,
            constraints=None,
            force_words_ids=None,
        )
        end = time.time()
        times.append(1000*(end - start))
    return times

def measure_time_onnx(model, batches, iterations):
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model.generate(batches)
        end = time.time()
        times.append(1000*(end - start))
    return times

if __name__ == '__main__':
    tokenizer = MarianTokenizer.from_pretrained(NAME)
    model_torch = MarianMTModel.from_pretrained(NAME)
    model_onnx = TranslationModelOnnx(model_torch.config)

    sentences = re.split(REGEX, TEXT)
    batches = tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")

    warm_up(model_torch, model_onnx, batches)
    torch_times = measure_time_torch(model_torch, batches, ITERATIONS)
    onnx_times = measure_time_onnx(model_onnx, batches, ITERATIONS)

    print(f"Torch latency: {np.around(np.mean(torch_times), 2)} ms")
    print(f"ONNX latency: {np.around(np.mean(onnx_times), 2)} ms")
