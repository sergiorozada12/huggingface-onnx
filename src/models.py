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
        return self.encoder_session.run(None, onnx_inputs)


class TranslationDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_session = InferenceSession("onnx/decoder.onnx")
        self.lm_head_session = InferenceSession("onnx/lm_head.onnx")

    def forward(self, input_ids, attention_mask, encoder_outputs, encoder_attention_mask, index):
        decoder_onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy(),
            'encoder_hidden_states': encoder_outputs.numpy(),
            'encoder_attention_mask': encoder_attention_mask.numpy()
        }
        decoder_onnx_outputs = self.decoder_session.run(None, decoder_onnx_inputs)
        hidden = torch.tensor(decoder_onnx_outputs[0], dtype=torch.float32)

        _, n_length, _ = hidden.shape
        mask_selection = torch.arange(n_length, dtype=torch.float32) == index
        mask_selection = mask_selection.view(1, -1, 1)
        masked_selection = torch.multiply(hidden, mask_selection)
        summed = torch.sum(masked_selection, 1)
        hidden_masked = torch.unsqueeze(summed, 1)

        print(hidden_masked.shape)

        lm_head_onnx_inputs = {'input': hidden_masked.numpy()}
        return self.lm_head_session.run(None, lm_head_onnx_inputs)


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
