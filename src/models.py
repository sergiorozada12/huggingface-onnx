import re
import numpy as np
from transformers import MarianMTModel, MarianTokenizer

from onnxruntime import InferenceSession


class TranslatorTorch():
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
        batches_translated = self.model.generate(
            **batches,
            num_beams=1,
            num_beam_groups=1,
            do_sample=False,
            constraints=None,
            force_words_ids=None,
        )
        print(batches_translated)
        sentences_translated = [self.tokenizer.decode(s, skip_special_tokens=True) for s in batches_translated]
        return " ".join(sentences_translated)
    
    def save_model(self, name):
        self.tokenizer.save_pretrained(name)
        self.model.save_pretrained(name)


class TranslationEncoderOnnx:
    def __init__(self):
        super().__init__()
        self.encoder_session = InferenceSession("onnx/encoder.opt.quant.onnx", providers=['CPUExecutionProvider'])

    def __call__(self, input_ids, attention_mask):
        onnx_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        hidden = self.encoder_session.run(None, onnx_inputs)
        return hidden[0]


class TranslationDecoderOnnx:
    def __init__(self):
        super().__init__()
        self.decoder_session = InferenceSession("onnx/decoder.opt.quant.onnx", providers=['CPUExecutionProvider'])
        self.decoder_pkv_session = InferenceSession("onnx/decoder_pkv.opt.quant.onnx", providers=['CPUExecutionProvider'])
        self.lm_head_session = InferenceSession("onnx/lm_head.opt.quant.onnx", providers=['CPUExecutionProvider'])

    def __call__(self, input_ids, encoder_outputs, encoder_attention_mask, past_key_values=None):
        if past_key_values:
            decoder_names = ['input_ids', 'encoder_attention_mask']
            decoder_inputs = [input_ids, encoder_attention_mask]
            decoder_onnx_inputs = dict(zip(decoder_names + self.pkv_names, decoder_inputs + past_key_values))

            output = self.decoder_pkv_session.run(None, decoder_onnx_inputs)
            hidden = output[0]
            pkv = output[1:]
        else:
            decoder_onnx_inputs = {
                'input_ids': input_ids,
                'encoder_hidden_states': encoder_outputs,
                'encoder_attention_mask': encoder_attention_mask
            }
            output = self.decoder_session.run(None, decoder_onnx_inputs)
            hidden = output[0]
            pkv = output[1:]

            self.pkv_names = [f"pkv_{i}" for i in range(len(pkv))]

        summed = np.sum(hidden, 1)
        hidden_masked = np.expand_dims(summed, 1)

        lm_head_onnx_inputs = {'input': hidden_masked}
        output = self.lm_head_session.run(None, lm_head_onnx_inputs)
        return output[0], pkv


class TranslationModelOnnx:
    def __init__(self, config, max_length=100):
        self.encoder = TranslationEncoderOnnx()
        self.decoder = TranslationDecoderOnnx()
        self.config = config
        self.max_length = max_length

    def generate(self, tokens):
        enc_inputs, enc_att_mask = tokens['input_ids'].numpy(), tokens['attention_mask'].numpy()
        hidden = self.encoder(enc_inputs, enc_att_mask)

        bsz = enc_inputs.shape[0]

        indices_active = np.arange(enc_inputs.shape[0])
        dec_inputs = np.ones((bsz, self.max_length), int)*self.config.pad_token_id
        dec_inputs[:, 0] = self.config.decoder_start_token_id

        past_key_values = None

        for idx in range(self.max_length - 1):
            if past_key_values:
                logits, past_key_values = self.decoder(
                    dec_inputs[indices_active, idx].reshape(-1, 1),
                    hidden[indices_active, :],
                    enc_att_mask[indices_active, :],
                    past_key_values
                )
            else:
                logits, past_key_values = self.decoder(
                    dec_inputs[indices_active, idx].reshape(-1, 1),
                    hidden[indices_active, :],
                    enc_att_mask[indices_active, :],
                )
            logits[:, 0, self.config.pad_token_id] = float("-inf")
            token_ids = logits.argmax(axis=2).flatten()
            dec_inputs[indices_active, idx + 1] = token_ids

            indices_non_end = np.where(token_ids != self.config.eos_token_id)[0]
            indices_active = indices_active[indices_non_end]

            if indices_active.size == 0:
                break

            past_key_values = [pkv[indices_non_end, :, :, :] for pkv in past_key_values]
        return dec_inputs[:, :idx + 2]


class TranslatorOnnx():
    def __init__(self, name, split_regex):
        self.split_regex = split_regex
        self.tokenizer = MarianTokenizer.from_pretrained(name)

        config_file = MarianMTModel.from_pretrained(name).config
        self.model = TranslationModelOnnx(config_file)
    
    def _prepare_text(self, text):
        text_filtered = text.replace('\n', ' ').strip()
        return re.split(self.split_regex, text_filtered)

    def translate(self, text):
        sentences = self._prepare_text(text)
        batches = self.tokenizer.prepare_seq2seq_batch(sentences, return_tensors="pt")
        batches_translated = self.model.generate(batches)
        sentences_translated = [self.tokenizer.decode(s, skip_special_tokens=True) for s in batches_translated]
        return " ".join(sentences_translated)
