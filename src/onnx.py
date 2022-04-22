import numpy as np

import torch
from transformers import MarianMTModel

import onnx
import onnxruntime
from onnxruntime.quantization import QuantizationMode, quantize

from src.wrappers import MarianDecoderWrapped


class OnnxConverter:
    def __init__(self, name, batch_size, max_length, embedding_size):
        self.model = MarianMTModel.from_pretrained(name)
        self.encoder = self.model.model.encoder
        self.decoder = MarianDecoderWrapped(self.model.model.decoder)

        lm_head_input_size = self.model.lm_head.weight.shape[0]
        lm_head_output_size = self.model.lm_head.weight.shape[1]
        self.lm_head = torch.nn.Linear(lm_head_input_size, lm_head_output_size, bias=True)
        self.lm_head.weight.data = self.model.lm_head.weight
        self.lm_head.bias.data = self.model.final_logits_bias

        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_size = embedding_size

    def _convert_encoder(self):
        encoder_input = torch.randint(10_000, (self.batch_size, self.max_length), requires_grad=False)
        padding_mask = torch.randint(1, (self.batch_size, self.max_length), requires_grad=False)
        encoder_hidden_state = self.encoder(encoder_input, padding_mask, return_dict=False)

        torch.onnx.export(
            self.encoder,
            (encoder_input, padding_mask),
            "onnx/encoder.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names = ['input_ids', 'attention_mask'],
            output_names = ['output'],
            dynamic_axes={
                'input_ids' : {0 : 'batch_size', 1: 'seq_length'},
                'attention_mask' : {0 : 'batch_size', 1: 'seq_length'},
                'output' : {0 : 'batch_size', 1: 'seq_length'}})

        onnx_session = onnxruntime.InferenceSession("onnx/encoder.onnx")
        onnx_inputs = {
            'input_ids': encoder_input.numpy(),
            'attention_mask': padding_mask.numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)

        np.testing.assert_allclose(encoder_hidden_state[0].detach().numpy(), onnx_outputs[0], rtol=1e-03, atol=1e-05)
        print("Encoder exported OK!")

    def _convert_decoder(self):
        decoder_input = torch.randint(10_000, (self.batch_size, self.max_length), requires_grad=False)
        encoder_hidden_states = torch.rand(self.batch_size, self.max_length, self.embedding_size, requires_grad=False)
        encoder_mask = torch.randint(1, (self.batch_size, self.max_length), requires_grad=False)
        decoder_hidden_states = self.decoder(decoder_input, encoder_hidden_states, encoder_mask)

        torch.onnx.export(
            self.decoder,
            (decoder_input, encoder_hidden_states, encoder_mask),
            "onnx/decoder.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names = ['input_ids', 'encoder_hidden_states', 'encoder_attention_mask'],
            output_names = ['output'],
            dynamic_axes={
                'input_ids' : {0 : 'batch_size', 1: 'seq_length'},
                'encoder_hidden_states' : {0 : 'batch_size', 1: 'seq_length'},
                'encoder_attention_mask' : {0 : 'batch_size', 1: 'seq_length'},
                'output' : {0 : 'batch_size', 1: 'seq_length'}})

        onnx_session = onnxruntime.InferenceSession("onnx/decoder.onnx")
        onnx_inputs = {
            'input_ids': decoder_input.numpy(),
            'encoder_hidden_states': encoder_hidden_states.numpy(),
            'encoder_attention_mask': encoder_mask.numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)

        np.testing.assert_allclose(decoder_hidden_states[0].detach().numpy(), onnx_outputs[0], rtol=1e-03, atol=1e-05)
        print("Decoder exported OK!")

    def _convert_lm_head(self):
        lm_head_input = torch.rand(self.batch_size, 1, self.embedding_size, requires_grad=False)
        lm_head_output = self.lm_head(lm_head_input)

        torch.onnx.export(
            self.lm_head,
            lm_head_input,
            "onnx/lm_head.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes={
                'input' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}})

        onnx_session = onnxruntime.InferenceSession("onnx/lm_head.onnx")
        onnx_inputs = {'input': lm_head_input.numpy()}
        onnx_outputs = onnx_session.run(None, onnx_inputs)

        np.testing.assert_allclose(lm_head_output.detach().numpy(), onnx_outputs[0], rtol=1e-03, atol=1e-05)
        print("LM Head exported OK!")

    def convert_to_onnx(self):
        self._convert_encoder()
        self._convert_decoder()
        self._convert_lm_head()


    def optimize_onnx_model(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess_options.optimized_model_filepath = "onnx/encoder.opt.onnx"
        _ = onnxruntime.InferenceSession("onnx/encoder.onnx", sess_options)

        sess_options.optimized_model_filepath = "onnx/decoder.opt.onnx"
        _ = onnxruntime.InferenceSession("onnx/decoder.onnx", sess_options)

        sess_options.optimized_model_filepath = "onnx/lm_head.opt.onnx"
        _ = onnxruntime.InferenceSession("onnx/lm_head.onnx", sess_options)
    
    def quantize_onnx_model(self):
        encoder = onnx.load("onnx/encoder.opt.onnx")
        decoder = onnx.load("onnx/decoder.opt.onnx")
        lm_head = onnx.load("onnx/lm_head.opt.onnx")

        encoder_quant = quantize(
            model=encoder,
            quantization_mode=QuantizationMode.IntegerOps,
            force_fusions=True,
            symmetric_weight=True,
        )

        decoder_quant = quantize(
            model=decoder,
            quantization_mode=QuantizationMode.IntegerOps,
            force_fusions=True,
            symmetric_weight=True,
        )

        lm_head_quant = quantize(
            model=lm_head,
            quantization_mode=QuantizationMode.IntegerOps,
            force_fusions=True,
            symmetric_weight=True,
        )

        onnx.save_model(encoder_quant, "onnx/encoder.opt.quant.onnx")
        onnx.save_model(decoder_quant, "onnx/decoder.opt.quant.onnx")
        onnx.save_model(lm_head_quant, "onnx/lm_head.opt.quant.onnx")
