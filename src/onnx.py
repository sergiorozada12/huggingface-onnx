from pathlib import Path
import numpy as np
import onnx
import onnxruntime

from transformers.onnx.convert import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers import MarianMTModel

import torch

class OnnxConverter:
    def __init__(self, name, batch_size, max_length, embedding_size):
        self.model = MarianMTModel.from_pretrained(name)
        self.encoder = self.model.model.encoder
        self.decoder = self.model.model.decoder

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
            opset_version=10,
            do_constant_folding=True,
            input_names = ['input_ids', 'attention_mask'],
            output_names = ['output'],
            dynamic_axes={
                'input_ids' : {0 : 'batch_size'},
                'attention_mask' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}})

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
        decoder_mask = torch.randint(1, (self.batch_size, self.max_length), requires_grad=False)
        encoder_hidden_states = torch.rand(self.batch_size, self.max_length, self.embedding_size, requires_grad=False)
        encoder_mask = torch.randint(1, (self.batch_size, self.max_length), requires_grad=False)
        decoder_hidden_states = self.decoder(decoder_input, decoder_mask, encoder_hidden_states, encoder_mask, return_dict=False)

        torch.onnx.export(
            self.decoder,
            (decoder_input, decoder_mask, encoder_hidden_states, encoder_mask),
            "onnx/decoder.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names = ['input_ids', 'attention_mask', 'encoder_hidden_states', 'encoder_attention_mask'],
            output_names = ['output'],
            dynamic_axes={
                'input_ids' : {0 : 'batch_size'},
                'attention_mask' : {0 : 'batch_size'},
                'encoder_hidden_states' : {0 : 'batch_size'},
                'encoder_attention_mask' : {0 : 'batch_size'},
                'output' : {0 : 'batch_size'}})

        onnx_session = onnxruntime.InferenceSession("onnx/decoder.onnx")
        onnx_inputs = {
            'input_ids': decoder_input.numpy(),
            'attention_mask': decoder_mask.numpy(),
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
            opset_version=10,
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


    def optimize_onnx_model():
        from onnxruntime_tools import optimizer
        from onnxruntime_tools.transformers.onnx_model_bert import BertOptimizationOptions

        # disable embedding layer norm optimization for better model size reduction
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_embed_layer_norm = False

        opt_model = optimizer.optimize_model(
            'onnx/bert-base-cased.onnx',
            'bert', 
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options)
        opt_model.save_model_to_file('bert.opt.onnx')
