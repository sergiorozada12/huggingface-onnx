import re
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from onnxruntime import InferenceSession

from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel


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
        self.main_input_name = 'encoder'

    def forward(self, input_ids, attention_mask):
        onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy()
        }
        hidden = self.encoder_session.run(None, onnx_inputs)
        return torch.tensor(hidden[0], dtype=torch.float32)


class TranslationDecoder(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        self.decoder_session = InferenceSession("onnx/decoder.onnx")
        self.lm_head_session = InferenceSession("onnx/lm_head.onnx")

        self.max_length = max_length
        self.main_input_name = 'decoder'

    def forward(self, input_ids, encoder_outputs, encoder_attention_mask, index):
        attention_maxk = [1]*(index + 1) + [0]*(self.max_length - index - 1)
        attention_maxk = torch.tensor([attention_maxk], dtype=torch.int64)

        decoder_onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_maxk.numpy(),
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

        lm_head_onnx_inputs = {'input': hidden_masked.numpy()}
        output = self.lm_head_session.run(None, lm_head_onnx_inputs)
        return torch.tensor(output[0], dtype=torch.float32)


class TranslationModelOnnx:
    def __init__(self, max_length, config):
        self.encoder = TranslationEncoder()
        self.decoder = TranslationDecoder(max_length)

        self.max_length = max_length
        self.config = config

    def generate(self, tokens):
        enc_inputs, enc_att_mask = tokens['input_ids'], tokens['attention_mask']
        hidden = self.encoder(enc_inputs, enc_att_mask)
        dec_inputs = [self.config.pad_token_id]*self.max_length
        dec_inputs[0] = self.config.decoder_start_token_id
        dec_inputs = torch.tensor([dec_inputs], dtype=torch.int64)
        print(dec_inputs)
        for idx in range(self.max_length):
            output = self.decoder(dec_inputs, hidden, enc_att_mask, idx)
            token_id = output.argmax().item()
            dec_inputs[0, idx] = token_id
            print(token_id)
            #print(output[idx])

