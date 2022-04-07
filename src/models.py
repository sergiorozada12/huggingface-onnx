import re
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from onnxruntime import InferenceSession


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
    
    def save_model(self, name):
        self.tokenizer.save_pretrained(name)
        self.model.save_pretrained(name)


class TranslationEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_session = InferenceSession("onnx/encoder.onnx")

    def forward(self, input_ids, attention_mask):
        onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy()
        }
        return self.encoder.run(None, onnx_inputs)


class TranslationDecoder(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.decoder = InferenceSession("onnx/decoder.onnx")
        self.lm_head = InferenceSession("onnx/lm_head.onnx")

        self.max_length = max_length

    def forward(self, input_ids, attention_mask, encoder_outputs, encoder_attention_mask, index):
        mask = np.triu(np.ones((self.max_length, self.max_length)), 1)
        mask[mask == 1] = -np.inf
        causal_mask = torch.tensor(mask, dtype=torch.float)

        onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
            'encoder_hidden_states': encoder_outputs.numpy(),
            'encoder_attention_mask': encoder_attention_mask.numpy()
        }
        onnx_outputs = self.decoder.run(None, onnx_inputs)
        hidden = onnx_outputs[0]

        _, n_length, _ = hidden.shape
        mask_selection = torch.arange(n_length, dtype=torch.float32) == index
        mask_selection = mask_selection.view(1, -1, 1)
        masked_selection = torch.multiply(hidden, mask_selection)
        summed = torch.sum(masked_selection, 1)
        hidden_masked = torch.unsqueeze(summed, 1)

        return torch.nn.functional.linear(
            input=hidden_masked,
            weight=self.lm_weights,
            bias=self.lm_bias
        )


class TranslationModelOnnx():
    def __init__(self, name_tokenizer, name_model, split_regex):
        self.split_regex = split_regex
        self.tokenizer = MarianTokenizer.from_pretrained(name_tokenizer)
        self.session = InferenceSession(name_model)

    def _prepare_text(self, text):
        text_filtered = text.replace('\n', ' ').strip()
        return re.split(self.split_regex, text_filtered)

    def translate(self, text):
        sentences = self._prepare_text(text)
        batches = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")
        batches['']
        batches_translated = self.session.run(output_names=["last_hidden_state"], input_feed=dict(batches))
        sentences_translated = [self.tokenizer.decode(s, skip_special_tokens=True) for s in batches_translated]
        return " ".join(sentences_translated)
